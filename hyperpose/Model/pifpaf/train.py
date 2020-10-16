#!/usr/bin/env python3

import math
import multiprocessing
import os
import cv2
import time
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import tensorlayer as tl
from pycocotools.coco import maskUtils
import _pickle as cPickle
from functools import partial
from .utils import get_pifmap,get_pafmap,draw_result,maps_to_numpy
from ..common import log,KUNGFU,MODEL,get_optim,init_log
from ..domainadapt import get_discriminator
from ..metrics import AvgMetric

#TODO:check all the x, y and scale correspond to the map shape(e.g. whether multiple by stride)
def regulize_loss(target_model,weight_decay_factor):
    re_loss=0
    regularizer=tf.keras.regularizers.l2(l=weight_decay_factor)
    for weight in target_model.trainable_weights:
        re_loss+=regularizer(weight)
    return re_loss

def _data_aug_fn(image, ground_truth, hin, hout, win, wout, parts, limbs ,flip_list=None, data_format="channels_first"):
    """Data augmentation function."""
    #restore data
    ground_truth = cPickle.loads(ground_truth.numpy())
    image=image.numpy()
    annos = ground_truth["kpt"]
    labeled= ground_truth["labeled"]
    mask = ground_truth["mask"]

    # decode mask
    h_mask, w_mask, _ = np.shape(image)
    mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)
    if(mask!=None):
        for seg in mask:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            mask_miss = np.bitwise_and(mask_miss, bin_mask)
    
    #get transform matrix
    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
    M_combined = M_rotate.dot(M_zoom)
    h, w, _ = image.shape
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
    
    #apply data augmentation
    image = tl.prepro.affine_transform_cv2(image, transform_matrix)
    mask_miss = tl.prepro.affine_transform_cv2(mask_miss, transform_matrix, border_mode='replicate')
    annos = tl.prepro.affine_transform_keypoints(annos, transform_matrix)
    #temply ignore flip augmentation 
    '''
    if(flip_list!=None):
        image, annos, mask_miss = tl.prepro.keypoint_random_flip(image,annos, mask_miss, prob=0.5, flip_list=flip_list)
    '''
    image, annos, mask_miss = tl.prepro.keypoint_resize_random_crop(image, annos, mask_miss, size=(hin, win)) # hao add

    # generate result incluing pif_maps and paf_maps
    height, width, _ = image.shape
    out_mask_miss=cv2.resize(mask_miss,(wout,hout))
    pif_conf,pif_vec,pif_scale = get_pifmap(annos, out_mask_miss, height, width, hout, wout, parts, limbs, data_format=data_format)
    paf_conf,paf_src_vec,paf_dst_vec,paf_src_scale,paf_dst_scale = get_pafmap(annos, out_mask_miss, height, width, hout, wout, parts, limbs, data_format=data_format)

    image=cv2.resize(image,(win,hin))
    mask_miss=cv2.resize(mask_miss,(win,hin))
    img_mask=mask_miss
    
    #generate output masked image, result map and maskes
    img_mask = mask_miss.reshape(hin, win, 1)
    image = image * np.repeat(img_mask, 3, 2)
    mask_miss = np.array(cv2.resize(mask_miss, (wout, hout), interpolation=cv2.INTER_AREA),dtype=np.float32)[:,:,np.newaxis]
    if(data_format=="channels_first"):
        image=np.transpose(image,[2,0,1])
        mask_miss=np.transpose(mask_miss,[2,0,1])
    labeled=np.float32(labeled)
    return image, pif_conf,pif_vec,pif_scale,paf_conf,paf_src_vec,paf_dst_vec,paf_src_scale,paf_dst_scale, mask_miss, labeled
    
def _map_fn(img_list, annos ,data_aug_fn, hin, win, hout, wout, parts, limbs):
    """TF Dataset pipeline."""
    #load data
    image = tf.io.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #data augmentation using affine transform and get paf maps
    image, pif_conf, pif_vec, pif_scale, paf_conf, paf_src_vec, paf_dst_vec, paf_src_scale, paf_dst_scale, mask, labeled = tf.py_function(\
        data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,\
            tf.float32,tf.float32,tf.float32, tf.float32, tf.float32])
    #data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=35./255.)   # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image, pif_conf, pif_vec, pif_scale, paf_conf, paf_src_vec, paf_dst_vec, paf_src_scale, paf_dst_scale, mask, labeled

def get_paramed_map_fn(hin,win,hout,wout,parts,limbs,flip_list=None,data_format="channels_first"):
    paramed_data_aug_fn=partial(_data_aug_fn,hin=hin,win=win,hout=hout,wout=wout,parts=parts,limbs=limbs,\
        flip_list=flip_list,data_format=data_format)
    paramed_map_fn=partial(_map_fn,data_aug_fn=paramed_data_aug_fn, hin=hin, win=win, hout=hout, wout=wout ,parts=parts, limbs=limbs)
    return paramed_map_fn

def single_train(train_model,dataset,config):
    '''Single train pipeline of Openpose class models

    input model and dataset, the train pipeline will start automaticly
    the train pipeline will:
    1.store and restore ckpt in directory ./save_dir/model_name/model_dir
    2.log loss information in directory ./save_dir/model_name/log.txt
    3.visualize model output periodly during training in directory ./save_dir/model_name/train_vis_dir
    the newest model is at path ./save_dir/model_name/model_dir/newest_model.npz

    Parameters
    ----------
    arg1 : tensorlayer.models.MODEL
        a preset or user defined model object, obtained by Model.get_model() function
    
    arg2 : dataset
        a constructed dataset object, obtained by Dataset.get_dataset() function
    
    
    Returns
    -------
    None
    '''

    init_log(config)
    #train hyper params
    #dataset params
    n_step = config.train.n_step
    batch_size = config.train.batch_size
    #learning rate params
    lr_init = config.train.lr_init
    lr_decay_factor = config.train.lr_decay_factor
    lr_decay_steps= [200000,300000,360000,420000,480000,540000,600000,700000,800000,900000]
    weight_decay_factor = config.train.weight_decay_factor
    #log and checkpoint params
    log_interval=config.log.log_interval
    save_interval=config.train.save_interval
    vis_dir=config.train.vis_dir
    
    #model hyper params
    hin = train_model.hin
    win = train_model.win
    hout = train_model.hout
    wout = train_model.wout
    model_dir = config.model.model_dir
    pretrain_model_dir=config.pretrain.pretrain_model_dir
    pretrain_model_path=f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"

    log(f"single training using learning rate:{lr_init} batch_size:{batch_size}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    parts,limbs,data_format=train_model.parts,train_model.limbs,train_model.data_format
    paramed_map_fn=get_paramed_map_fn(hin,win,hout,wout,parts,limbs,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    train_dataset = train_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    train_dataset = train_dataset.batch(batch_size)  
    train_dataset = train_dataset.prefetch(64)
    
    #train configure
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    lr_init=tf.Variable(lr_init,trainable=False)
    opt=tf.optimizers.SGD(learning_rate=lr,momentum=0.95)
    ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr)
    ckpt_manager=tf.train.CheckpointManager(ckpt,model_dir,max_to_keep=3)
    
    #load from ckpt
    log("loading ckpt...")
    try:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log("ckpt loaded successfully!")
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    #load pretrained backbone
    log("loading pretrained backbone...")
    if(tl.files.load_and_assign_npz_dict(name=pretrain_model_path,network=train_model.backbone,skip=True)):
        log("pretrained backbone loaded successfully")
    else:
        log("pretrained backbone doesn't exist, model backbone are initialized")
    #load model weights
    log("loading saved training model weights...")
    try:
        train_model.load_weights(os.path.join(model_dir,"newest_model.npz"))
        log("saved training model weights loaded successfully")
    except:
        log("model_path doesn't exist, model parameters are initialized")
    
    for lr_decay_step in lr_decay_steps:
        if(step>lr_decay_step):
            lr=lr*lr_decay_factor
        
    #optimize one step
    def one_step(image,gt_label,mask,train_model):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            gt_pif_maps,gt_paf_maps=gt_label
            pd_pif_maps,pd_paf_maps=train_model.forward(image,is_train=True)
            loss_pif_maps,loss_paf_maps=train_model.cal_loss(pd_pif_maps,pd_paf_maps,gt_pif_maps,gt_paf_maps)
            total_loss=sum(loss_pif_maps)+sum(loss_paf_maps)
        
        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        return pd_pif_maps,pd_paf_maps,loss_pif_maps,loss_paf_maps,total_loss

    #train each step
    train_model.train()
    tic=time.time()
    avg_time=AvgMetric(name="time_iter",metric_interval=log_interval)
    #total loss metrics
    avg_total_loss=AvgMetric(name="total_loss",metric_interval=log_interval)
    #pif loss metrics
    avg_pif_conf_loss=AvgMetric(name="pif_conf_loss",metric_interval=log_interval)
    avg_pif_vec_loss=AvgMetric(name="pif_vec_loss",metric_interval=log_interval)
    avg_pif_scale_loss=AvgMetric(name="pif_scale_loss",metric_interval=log_interval)
    #paf loss metrics
    avg_paf_conf_loss=AvgMetric(name="paf_conf_loss",metric_interval=log_interval)
    avg_paf_src_vec_loss=AvgMetric(name="paf_src_vec_loss",metric_interval=log_interval)
    avg_paf_dst_vec_loss=AvgMetric(name="paf_dst_vec_loss",metric_interval=log_interval)
    avg_paf_src_scale_loss=AvgMetric(name="paf_src_scale_loss",metric_interval=log_interval)
    avg_paf_dst_scale_loss=AvgMetric(name="paf_dst_scale_loss",metric_interval=log_interval)
    log('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_steps: {} lr_decay_factor: {} weight_decay_factor: {}'.format(
            n_step, batch_size, lr_init.numpy(), lr_decay_steps, lr_decay_factor, weight_decay_factor))
    for image, pif_conf, pif_vec, pif_scale, paf_conf, paf_src_vec, paf_dst_vec, paf_src_scale, paf_dst_scale, mask, labeled in train_dataset:
        #get losses
        #debug
        gt_pif_maps=[pif_conf,pif_vec,pif_scale]
        gt_paf_maps=[paf_conf,paf_src_vec,paf_dst_vec,paf_src_scale,paf_dst_scale]
        gt_label=[gt_pif_maps,gt_paf_maps]
        pd_pif_maps,pd_paf_maps,loss_pif_maps,loss_paf_maps,total_loss=one_step(image,gt_label,mask,train_model)
        loss_pif_conf,loss_pif_vec,loss_pif_scale=loss_pif_maps
        loss_paf_conf,loss_paf_src_vec,loss_paf_dst_vec,loss_paf_src_scale,loss_paf_dst_scale=loss_paf_maps
        #update metrics
        avg_time.update(time.time()-tic)
        tic=time.time()
        #update total losses
        avg_total_loss.update(total_loss)
        #update pif_losses metrics
        avg_pif_conf_loss.update(loss_pif_conf)
        avg_pif_vec_loss.update(loss_pif_vec)
        avg_pif_scale_loss.update(loss_pif_scale)
        #update paf_losses metrics
        avg_paf_conf_loss.update(loss_paf_conf)
        avg_paf_src_vec_loss.update(loss_paf_src_vec)
        avg_paf_dst_vec_loss.update(loss_paf_dst_vec)
        avg_paf_src_scale_loss.update(loss_paf_src_scale)
        avg_paf_dst_scale_loss.update(loss_paf_dst_scale)

        #learning rate decay
        if(step in lr_decay_steps):
            new_lr_decay = lr_decay_factor**(lr_decay_steps.index(step)+1) 
            lr=lr_init*new_lr_decay

        #save log info periodly
        if((step.numpy()!=0) and (step.numpy()%log_interval)==0):
            log(f"Train iteration {step.numpy()} / {n_step}, Learning rate:{lr.numpy()} {avg_total_loss.get_metric()} "+\
                f"{avg_pif_conf_loss.get_metric()} {avg_pif_vec_loss.get_metric()} {avg_pif_scale_loss.get_metric()} "+\
                f"{avg_paf_conf_loss.get_metric()} {avg_paf_src_vec_loss.get_metric()} {avg_paf_dst_vec_loss.get_metric()} "+\
                f"{avg_paf_src_scale_loss.get_metric()} {avg_paf_dst_scale_loss.get_metric()} {avg_time.get_metric()} ")

        #save result and ckpt periodly
        if(step.numpy()!=0 and step.numpy()%save_interval==0):
            #save ckpt
            log("saving model ckpt and result...")
            ckpt_save_path=ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            #save train model
            model_save_path=os.path.join(model_dir,"newest_model.npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")
            #draw result
            stride=train_model.stride
            gt_pif_maps,gt_paf_maps=gt_label
            #turn into numpy
            pd_pif_maps=maps_to_numpy(pd_pif_maps)
            pd_paf_maps=maps_to_numpy(pd_paf_maps)
            gt_pif_maps=maps_to_numpy(gt_pif_maps)
            gt_paf_maps=maps_to_numpy(gt_paf_maps)
            draw_result(image.numpy(),pd_pif_maps,pd_paf_maps,gt_pif_maps,gt_paf_maps,mask.numpy(),parts,limbs,stride,save_dir=vis_dir,\
                name=f"train_{step.numpy()}")

        #training finished
        if(step==n_step):
            break

def parallel_train(train_model,dataset,config):
    '''Parallel train pipeline of openpose class models

    input model and dataset, the train pipeline will start automaticly
    the train pipeline will:
    1.store and restore ckpt in directory ./save_dir/model_name/model_dir
    2.log loss information in directory ./save_dir/model_name/log.txt
    3.visualize model output periodly during training in directory ./save_dir/model_name/train_vis_dir
    the newest model is at path ./save_dir/model_name/model_dir/newest_model.npz

    Parameters
    ----------
    arg1 : tensorlayer.models.MODEL
        a preset or user defined model object, obtained by Model.get_model() function
    
    arg2 : dataset
        a constructed dataset object, obtained by Dataset.get_dataset() function
    
    
    Returns
    -------
    None
    '''

    init_log(config)
    #train hyper params
    #dataset params
    n_step = config.train.n_step
    batch_size = config.train.batch_size
    #learning rate params
    lr_init = config.train.lr_init
    lr_decay_factor = config.train.lr_decay_factor
    lr_decay_steps= [200000,300000,360000,420000,480000,540000,600000,700000,800000,900000]
    weight_decay_factor = config.train.weight_decay_factor
    #log and checkpoint params
    log_interval=config.log.log_interval
    save_interval=config.train.save_interval
    vis_dir=config.train.vis_dir
    
    #model hyper params
    hin = train_model.hin
    win = train_model.win
    hout = train_model.hout
    wout = train_model.wout
    model_dir = config.model.model_dir
    pretrain_model_dir=config.pretrain.pretrain_model_dir
    pretrain_model_path=f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"

    #import kungfu
    from kungfu import current_cluster_size, current_rank
    from kungfu.tensorflow.initializer import broadcast_variables
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, SynchronousAveragingOptimizer, PairAveragingOptimizer

    log(f"parallel training using learning rate:{lr_init} batch_size:{batch_size}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    dataset_type=dataset.get_dataset_type()
    parts,limbs,data_format=train_model.parts,train_model.limbs,train_model.data_format
    paramed_map_fn=get_paramed_map_fn(hin,win,hout,wout,parts,limbs,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096)
    train_dataset = train_dataset.shard(num_shards=current_cluster_size(),index=current_rank())
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    train_dataset = train_dataset.batch(batch_size)  
    train_dataset = train_dataset.prefetch(64)
    
    #train configure
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    lr_init=tf.Variable(lr_init,trainable=False)
    opt=tf.optimizers.SGD(learning_rate=lr,momentum=0.95)
    ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr)
    ckpt_manager=tf.train.CheckpointManager(ckpt,model_dir,max_to_keep=3)
    
    #load from ckpt
    log("loading ckpt...")
    try:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log("ckpt loaded successfully!")
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")

    #load pretrained backbone
    log("loading pretrained backbone...")
    try:
        tl.files.load_and_assign_npz_dict(name=pretrain_model_path,network=train_model.backbone,skip=True)
        log("pretrained backbone loaded successfully")
    except:
        log("pretrained backbone doesn't exist, model backbone are initialized")

    #load model weights
    log("loading saved training model weights...")
    try:
        train_model.load_weights(os.path.join(model_dir,"newest_model.npz"))
        log("saved training model weights loaded successfully")
    except:
        log("model_path doesn't exist, model parameters are initialized")
    
    # KungFu configure
    kungfu_option=config.train.kungfu_option
    if kungfu_option == KUNGFU.Sync_sgd:
        print("using Kungfu.SynchronousSGDOptimizer!")
        opt = SynchronousSGDOptimizer(opt)
    elif kungfu_option == KUNGFU.Sync_avg:
        print("using Kungfu.SynchronousAveragingOptimize!")
        opt = SynchronousAveragingOptimizer(opt)
    elif kungfu_option == KUNGFU.Pair_avg:
        print("using Kungfu.PairAveragingOptimizer!")
        opt=PairAveragingOptimizer(opt)
    
    # KungFu adjust
    n_step = n_step // current_cluster_size() + 1  # KungFu
    for step_idx,step in enumerate(lr_decay_steps):
        lr_decay_steps[step_idx] = step // current_cluster_size() + 1  # KungFu

    for lr_decay_step in lr_decay_steps:
        if(step>lr_decay_step):
            lr=lr*lr_decay_factor
    
    #optimize one step
    @tf.function
    def one_step(image,gt_label,mask,train_model,is_first_batch=False):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            gt_pif_maps,gt_paf_maps=gt_label
            pd_pif_maps,pd_paf_maps=train_model.forward(image,is_train=True)
            loss_pif_maps,loss_paf_maps=train_model.cal_loss(pd_pif_maps,pd_paf_maps,gt_pif_maps,gt_paf_maps)
            total_loss=sum(loss_pif_maps)+sum(loss_paf_maps)
        
        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        #Kung fu
        if(is_first_batch):
            broadcast_variables(train_model.all_weights)
            broadcast_variables(opt.variables())
        return pd_pif_maps,pd_paf_maps,loss_pif_maps,loss_paf_maps,total_loss

    #train each step
    train_model.train()
    tic=time.time()
    avg_time=AvgMetric(name="time_iter",metric_interval=log_interval)
    #total loss metrics
    avg_total_loss=AvgMetric(name="total_loss",metric_interval=log_interval)
    #pif loss metrics
    avg_pif_conf_loss=AvgMetric(name="pif_conf_loss",metric_interval=log_interval)
    avg_pif_vec_loss=AvgMetric(name="pif_vec_loss",metric_interval=log_interval)
    avg_pif_scale_loss=AvgMetric(name="pif_scale_loss",metric_interval=log_interval)
    #paf loss metrics
    avg_paf_conf_loss=AvgMetric(name="paf_conf_loss",metric_interval=log_interval)
    avg_paf_src_vec_loss=AvgMetric(name="paf_src_vec_loss",metric_interval=log_interval)
    avg_paf_dst_vec_loss=AvgMetric(name="paf_dst_vec_loss",metric_interval=log_interval)
    avg_paf_src_scale_loss=AvgMetric(name="paf_src_scale_loss",metric_interval=log_interval)
    avg_paf_dst_scale_loss=AvgMetric(name="paf_dst_scale_loss",metric_interval=log_interval)
    log('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_steps: {} lr_decay_factor: {} weight_decay_factor: {}'.format(
            n_step, batch_size, lr_init.numpy(), lr_decay_steps, lr_decay_factor, weight_decay_factor))
    for image,gt_label,mask,labeled in train_dataset:
        #get losses
        pd_pif_maps,pd_paf_maps,loss_pif_maps,loss_paf_maps,total_loss=one_step(image,gt_label,mask,train_model,step==0)
        loss_pif_conf,loss_pif_vec,loss_pif_scale=loss_pif_maps
        loss_paf_conf,loss_paf_src_vec,loss_paf_dst_vec,loss_paf_src_scale,loss_paf_dst_scale=loss_paf_maps
        #update metrics
        avg_time.update(time.time()-tic)
        tic=time.time()
        #update total losses
        avg_total_loss.update(total_loss)
        #update pif_losses metrics
        avg_pif_conf_loss.update(loss_pif_conf)
        avg_pif_vec_loss.update(loss_pif_vec)
        avg_pif_scale_loss.update(loss_pif_scale)
        #update paf_losses metrics
        avg_paf_conf_loss.update(loss_paf_conf)
        avg_paf_src_vec_loss.update(loss_paf_src_vec)
        avg_paf_dst_vec_loss.update(loss_paf_dst_vec)
        avg_paf_src_scale_loss.update(loss_paf_src_scale)
        avg_paf_dst_scale_loss.update(loss_paf_dst_scale)

        #learning rate decay
        if(step in lr_decay_steps):
            new_lr_decay = lr_decay_factor**(lr_decay_steps.index(step)+1) 
            lr=lr_init*new_lr_decay

        #save log info periodly
        if((step.numpy()!=0) and (step.numpy()%log_interval)==0):
            log(f"Train iteration {n_step} / {step.numpy()}, Learning rate:{lr.numpy()} {avg_total_loss.get_metric()} "+\
                f"{avg_pif_conf_loss.get_metric()} {avg_pif_vec_loss.get_metric()} {avg_pif_scale_loss.get_metric()}"+\
                f"{avg_paf_conf_loss.get_metric()} {avg_paf_src_vec_loss.get_metric()} {avg_paf_dst_vec_loss.get_metric()}"+\
                f"{avg_paf_src_scale_loss.get_metric()} {avg_paf_dst_scale_loss.get_metric()} {avg_time.get_metric()}")

        #save result and ckpt periodly
        if((step.numpy()!=0) and (step.numpy()%save_interval)==0):
            #save ckpt
            log("saving model ckpt and result...")
            ckpt_save_path=ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            #save train model
            model_save_path=os.path.join(model_dir,"newest_model.npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")
            #draw result
            stride=train_model.stride
            gt_pif_maps,gt_paf_maps=gt_label
            draw_result(image,pd_pif_maps,pd_paf_maps,gt_pif_maps,gt_paf_maps,mask,parts,limbs,stride,save_dir=vis_dir,\
                name=f"train_{step.numpy()}")

        #training finished
        if(step==n_step):
            break