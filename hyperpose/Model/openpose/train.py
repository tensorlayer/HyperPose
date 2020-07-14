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
from .utils import tf_repeat, get_heatmap, get_vectormap, draw_results 
from .utils import get_parts,get_limbs,get_flip_list
from ..common import init_log,log,KUNGFU

def regulize_loss(target_model,weight_decay_factor):
    re_loss=0
    regularizer=tf.keras.regularizers.l2(l=weight_decay_factor)
    for trainable_weight in target_model.trainable_weights:
        re_loss+=regularizer(trainable_weight)
    return re_loss

def _data_aug_fn(image, ground_truth, hin, hout, win, wout, parts, limbs ,flip_list=None, data_format="channels_first"):
    """Data augmentation function."""
    #restore data
    concat_dim=0 if data_format=="channels_first" else -1
    ground_truth = cPickle.loads(ground_truth.numpy())
    image=image.numpy()
    annos = ground_truth["kpt"]
    mask = ground_truth["mask"]

    # decode mask
    h_mask, w_mask, _ = np.shape(image)
    mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)
    if(mask!=None):
        for seg in mask:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            mask_miss = np.bitwise_and(mask_miss, bin_mask)
    
    #debug no augmentation
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
    '''
    if(flip_list!=None):
        image, annos, mask_miss = tl.prepro.keypoint_random_flip(image,annos, mask_miss, prob=0.5, flip_list=flip_list)
    '''
    image, annos, mask_miss = tl.prepro.keypoint_resize_random_crop(image, annos, mask_miss, size=(hin, win)) # hao add

    # generate result which include keypoints heatmap and vectormap
    height, width, _ = image.shape
    heatmap = get_heatmap(annos, height, width, hout, wout, parts, limbs)
    vectormap = get_vectormap(annos, height, width, hout, wout, parts, limbs)
    resultmap = np.concatenate((heatmap, vectormap), axis=concat_dim)

    image=cv2.resize(image,(win,hin))
    mask_miss=cv2.resize(mask_miss,(win,hin))
    img_mask=mask_miss
    
    #generate output masked image, result map and maskes
    img_mask = mask_miss.reshape(hin, win, 1)
    image = image * np.repeat(img_mask, 3, 2)
    resultmap = np.array(resultmap, dtype=np.float32)
    mask_miss = np.array(cv2.resize(mask_miss, (wout, hout), interpolation=cv2.INTER_AREA),dtype=np.float32)[:,:,np.newaxis]
    if(data_format=="channels_first"):
        image=np.transpose(image,[2,0,1])
        mask_miss=np.transpose(mask_miss,[2,0,1])
    return image, resultmap, mask_miss


def _map_fn(img_list, annos ,data_aug_fn, hin, win, hout, wout, parts, limbs):
    """TF Dataset pipeline."""
    #load data
    image = tf.io.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #data augmentation using affine transform and get paf maps
    image, resultmap, mask = tf.py_function(data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32])
    #data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=45./255.)   # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image, resultmap, mask

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
    lr_decay_every_step = config.train.lr_decay_every_step
    weight_decay_factor = config.train.weight_decay_factor
    #log and checkpoint params
    log_interval=config.log.log_interval
    save_interval=config.train.save_interval
    vis_dir=config.train.vis_dir
    
    #model hyper params
    n_pos = train_model.n_pos
    hin = train_model.hin
    win = train_model.win
    hout = train_model.hout
    wout = train_model.wout
    model_dir = config.model.model_dir

    print(f"single training using learning rate:{lr_init} batch_size:{batch_size}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    dataset_type=dataset.get_dataset_type()
    parts,limbs,data_format=train_model.parts,train_model.limbs,train_model.data_format
    flip_list=get_flip_list(dataset_type)
    paramed_map_fn=get_paramed_map_fn(hin,win,hout,wout,parts,limbs,flip_list=flip_list,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    train_dataset = train_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    train_dataset = train_dataset.batch(config.train.batch_size)  
    train_dataset = train_dataset.prefetch(64)
    
    #train configure
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr)
    ckpt_manager=tf.train.CheckpointManager(ckpt,model_dir,max_to_keep=3)
    
    #load from ckpt
    try:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    try:
        train_model.load_weights(os.path.join(model_dir,"newest_model.npz"))
    except:
        log("model_path doesn't exist, model parameters are initialized")
        
    #optimize one step
    @tf.function
    def one_step(image,gt_label,mask,train_model):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            gt_conf=gt_label[:,:n_pos,:,:]
            gt_paf=gt_label[:,n_pos:,:,:]
            pd_conf,pd_paf,stage_confs,stage_pafs=train_model.forward(image,is_train=True)

            pd_loss,loss_confs,loss_pafs=train_model.cal_loss(gt_conf,gt_paf,mask,stage_confs,stage_pafs)
            re_loss=regulize_loss(train_model,weight_decay_factor)
            total_loss=pd_loss+re_loss
        
        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        return gt_conf,gt_paf,pd_conf,pd_paf,total_loss,re_loss,loss_confs,loss_pafs

    #train each step
    tic=time.time()
    train_model.train()
    log('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_every_step: {}'.format(
            n_step, batch_size, lr_init, lr_decay_every_step))
    for image,gt_label,mask in train_dataset:
        #learning rate decay
        if(step % lr_decay_every_step==0):
            new_lr_decay = lr_decay_factor**(step / lr_decay_every_step) 
            lr=lr_init*new_lr_decay
        #optimize one step
        gt_conf,gt_paf,pd_conf,pd_paf,total_loss,re_loss,loss_confs,loss_pafs=one_step(image.numpy(),gt_label.numpy(),mask.numpy(),train_model)
        #debug
        #for stage_id,(loss_conf,loss_paf) in enumerate(zip(loss_confs,loss_pafs)):
        #    print(f"test stage:{stage_id} loss_conf:{loss_conf} loss_paf:{loss_paf}")

        #save log info periodly
        if((step!=0) and (step%log_interval)==0):
            tic=time.time()
            log('Total Loss at iteration {} / {} is: {} Learning rate {} l2_loss {} time:{}'.format(
                step.numpy(), n_step, total_loss, lr.numpy(), re_loss,time.time()-tic))

        #save result and ckpt periodly
        if((step!=0) and (step%save_interval)==0):
            log("saving model ckpt and result...")
            draw_results(image.numpy(), gt_conf.numpy(), pd_conf.numpy(), gt_paf.numpy(), pd_paf.numpy(), mask.numpy(),\
                 vis_dir,'train_%d_' % step)
            ckpt_save_path=ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            model_save_path=os.path.join(model_dir,"newest_model.npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")

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
    lr_decay_every_step = config.train.lr_decay_every_step
    weight_decay_factor = config.train.weight_decay_factor
    #log and checkpoint params
    log_interval=config.log.log_interval
    save_interval=config.train.save_interval
    vis_dir=config.train.vis_dir
    
    #model hyper params
    n_pos = train_model.n_pos
    hin = train_model.hin
    win = train_model.win
    hout = train_model.hout
    wout = train_model.wout
    model_dir = config.model.model_dir

    #import kungfu
    from kungfu import current_cluster_size, current_rank
    from kungfu.tensorflow.initializer import broadcast_variables
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, SynchronousAveragingOptimizer, PairAveragingOptimizer
    

    print(f"parallel training using learning rate:{lr_init} batch_size:{batch_size}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    dataset_type=dataset.get_dataset_type()
    parts,limbs,data_format=train_model.parts,train_model.limbs,train_model.data_format
    flip_list=get_flip_list(dataset_type)
    paramed_map_fn=get_paramed_map_fn(hin,win,hout,wout,parts,limbs,flip_list=flip_list,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096)
    train_dataset = train_dataset.shard(num_shards=current_cluster_size(),index=current_rank())
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)  
    train_dataset = train_dataset.prefetch(64)

    #train model configure  
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    opt=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
    ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr)
    ckpt_manager=tf.train.CheckpointManager(ckpt,model_dir,max_to_keep=3)

    #load from ckpt
    try:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    try:
        train_model.load_weights(os.path.join(model_dir,"newest_model.npz"))
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
    

    n_step = n_step // current_cluster_size() + 1  # KungFu
    lr_decay_every_step = lr_decay_every_step // current_cluster_size() + 1  # KungFu
    
    #optimize one step
    @tf.function
    def one_step(image,gt_label,mask,train_model,is_first_batch=False):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            gt_conf=gt_label[:,:n_pos,:,:]
            gt_paf=gt_label[:,n_pos:,:,:]
            pd_conf,pd_paf,stage_confs,stage_pafs=train_model.forward(image,is_train=True)

            pd_loss,loss_confs,loss_pafs=train_model.cal_loss(gt_conf,gt_paf,mask,stage_confs,stage_pafs)
            re_loss=regulize_loss(train_model,weight_decay_factor)
            total_loss=pd_loss+re_loss
        
        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        #Kung fu
        if(is_first_batch):
            broadcast_variables(train_model.all_weights)
            broadcast_variables(opt.variables())
        return gt_conf,gt_paf,pd_conf,pd_paf,total_loss,re_loss

    #train each step
    tic=time.time()
    train_model.train()
    log(f"Worker {current_rank()}: Initialized")
    log('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_every_step: {}'.format(
            n_step, batch_size, lr_init, lr_decay_every_step))
    for image,gt_label,mask in train_dataset:
        #learning rate decay
        if(step % lr_decay_every_step==0):
            new_lr_decay = lr_decay_factor**(step // lr_decay_every_step)
            lr=lr_init*new_lr_decay
        #optimize one step
        gt_conf,gt_paf,pd_conf,pd_paf,total_loss,re_loss=one_step(image.numpy(),gt_label.numpy(),mask.numpy(),\
            train_model,step==0)
        #save log info periodly
        if((step!=0) and (step%log_interval)==0):
            tic=time.time()
            log('Total Loss at iteration {} / {} is: {} Learning rate {} l2_loss {} time:{}'.format(
                step.numpy(), n_step, total_loss, lr.numpy(), re_loss,time.time()-tic))

        #save result and ckpt periodly
        if((step!=0) and (step%save_interval)==0 and current_rank()==0):
            log("saving model ckpt and result...")
            draw_results(image.numpy(), gt_conf.numpy(), pd_conf.numpy(), gt_paf.numpy(), pd_paf.numpy(), mask.numpy(),\
                 vis_dir,'train_%d_' % step)
            ckpt_save_path=ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            model_save_path=os.path.join(model_dir,"newest_model.npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")

        #training finished
        if(step==n_step):
            break
