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
from .processor import PreProcessor
from .utils import tf_repeat, draw_results
from .utils import get_parts,get_limbs,get_flip_list
from ..augmentor import Augmentor
from ..common import log,KUNGFU,MODEL,get_optim,init_log,regulize_loss
from ..domainadapt import get_discriminator

def _data_aug_fn(image, ground_truth, augmentor, preprocessor, data_format="channels_first"):
    """Data augmentation function."""
    #restore data
    concat_dim=0 if data_format=="channels_first" else -1
    ground_truth = cPickle.loads(ground_truth.numpy())
    image=image.numpy()
    annos = ground_truth["kpt"]
    labeled= ground_truth["labeled"]
    mask = ground_truth["mask"]
    hin,win=preprocessor.hin,preprocessor.win
    hout,wout=preprocessor.hout,preprocessor.wout

    # decode mask
    h_mask, w_mask, _ = np.shape(image)
    mask_valid = np.ones((h_mask, w_mask), dtype=np.uint8)
    if(mask!=None):
        for seg in mask:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            if(bin_mask.shape!=mask_valid.shape):
                print(f"test error mask shape mask_valid:{mask_valid.shape} bin_mask:{bin_mask.shape}")
            else:
                mask_valid = np.bitwise_and(mask_valid, bin_mask)
    
    #general augmentaton process
    image,annos,mask_valid=augmentor.process(image=image,annos=annos,mask_valid=mask_valid)

    # generate result including heatmap and vectormap
    heatmap,vectormap=preprocessor.process(annos=annos,mask_valid=mask_valid)
    resultmap = np.concatenate((heatmap, vectormap), axis=concat_dim)
    
    #generate output masked image, result map and maskes
    image_mask = mask_valid.reshape(hin, win, 1)
    image = image * np.repeat(image_mask, 3, 2)
    resultmap = np.array(resultmap, dtype=np.float32)
    mask_valid = np.array(cv2.resize(mask_valid, (wout, hout), interpolation=cv2.INTER_AREA),dtype=np.float32)[:,:,np.newaxis]
    if(data_format=="channels_first"):
        image=np.transpose(image,[2,0,1])
        mask_valid=np.transpose(mask_valid,[2,0,1])
    labeled=np.float32(labeled)
    return image, resultmap, mask_valid, labeled


def _map_fn(img_list, annos ,data_aug_fn):
    """TF Dataset pipeline."""
    #load data
    image = tf.io.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #data augmentation using affine transform and get paf maps
    image, resultmap, mask, labeled = tf.py_function(data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32, tf.float32])
    #data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=35./255.)   # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image, resultmap, mask, labeled

def get_paramed_map_fn(augmentor,preprocessor,data_format="channels_first"):
    paramed_data_aug_fn=partial(_data_aug_fn,augmentor=augmentor,preprocessor=preprocessor,data_format=data_format)
    paramed_map_fn=partial(_map_fn,data_aug_fn=paramed_data_aug_fn)
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
    n_pos = train_model.n_pos
    hin = train_model.hin
    win = train_model.win
    hout = train_model.hout
    wout = train_model.wout
    parts,limbs,colors=train_model.parts,train_model.limbs,train_model.colors
    data_format=train_model.data_format
    model_dir = config.model.model_dir
    pretrain_model_dir=config.pretrain.pretrain_model_dir
    pretrain_model_path=f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"

    print(f"single training using learning rate:{lr_init} batch_size:{batch_size}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    augmentor=Augmentor(hin=hin,win=win,angle_min=-30,angle_max=30,zoom_min=0.5,zoom_max=0.8,flip_list=None)
    preprocessor=PreProcessor(parts=parts,limbs=limbs,hin=hin,win=win,hout=hout,wout=wout,colors=colors,data_format=data_format)
    paramed_map_fn=get_paramed_map_fn(augmentor=augmentor,preprocessor=preprocessor,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    train_dataset = train_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    train_dataset = train_dataset.batch(config.train.batch_size)  
    train_dataset = train_dataset.prefetch(64)
    
    #train configure
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    lr_init=tf.Variable(lr_init,trainable=False)
    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    #domain adaptation params
    domainadapt_flag=config.data.domainadapt_flag
    if(domainadapt_flag):
        print("domain adaptaion enabled!")
        discriminator=get_discriminator(train_model)
        opt_d=tf.keras.optimizers.Adam(learning_rate=lr)
        lambda_d=tf.Variable(1,trainable=False)
        ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr,optimizer_d=opt_d,lambda_d=lambda_d)
    else:
        ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr)
    ckpt_manager=tf.train.CheckpointManager(ckpt,model_dir,max_to_keep=3)
    
    #load from ckpt
    try:
        log("loading ckpt...")
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    #load pretrained backbone
    try:
        log("loading pretrained backbone...")
        tl.files.load_and_assign_npz_dict(name=pretrain_model_path,network=train_model.backbone,skip=True)
    except:
        log("pretrained backbone doesn't exist, model backbone are initialized")
    #load model weights
    try:
        log("loading saved training model weights...")
        train_model.load_weights(os.path.join(model_dir,"newest_model.npz"))
    except:
        log("model_path doesn't exist, model parameters are initialized")
    if(domainadapt_flag):
        try:
            log("loading saved domain adaptation discriminator weight...")
            discriminator.load_weights(os.path.join(model_dir,"newest_discriminator.npz"))
        except:
            log("discriminator path doesn't exist, discriminator parameters are initialized")
    
    for lr_decay_step in lr_decay_steps:
        if(step>lr_decay_step):
            lr=lr*lr_decay_factor
        
    #optimize one step
    @tf.function
    def one_step(image,gt_conf,gt_paf,mask,train_model):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            pd_conf,pd_paf,stage_confs,stage_pafs=train_model.forward(image,is_train=True)
            pd_loss,loss_confs,loss_pafs=train_model.cal_loss(gt_conf,gt_paf,mask,stage_confs,stage_pafs)
            re_loss=regulize_loss(train_model,weight_decay_factor)
            total_loss=pd_loss+re_loss
        
        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        return pd_conf,pd_paf,total_loss,re_loss,loss_confs,loss_pafs
    
    @tf.function
    def one_step_domainadpat(image,gt_conf,gt_paf,mask,labeled,train_model,discriminator,lambda_d):
        step.assign_add(1)
        with tf.GradientTape(persistent=True) as tape:
            #optimize train model
            pd_conf,pd_paf,stage_confs,stage_pafs,backbone_fatures=train_model.forward(image,is_train=True,domainadapt=True)
            d_predict=discriminator.forward(backbone_fatures)
            pd_loss,loss_confs,loss_pafs=train_model.cal_loss(gt_conf,gt_paf,mask,stage_confs,stage_pafs)
            re_loss=regulize_loss(train_model,weight_decay_factor)
            gan_loss=lambda_d*tf.nn.sigmoid_cross_entropy_with_logits(logits=d_predict,labels=1-labeled)
            total_loss=pd_loss+re_loss+gan_loss
            d_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=d_predict,labels=labeled)
        #optimize G
        g_gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(g_gradients,train_model.trainable_weights))
        #optimize D
        d_gradients=tape.gradient(d_loss,discriminator.trainable_weights)
        opt_d.apply_gradients(zip(d_gradients,discriminator.trainable_weights))
        return pd_conf,pd_paf,total_loss,re_loss,loss_confs,loss_pafs,gan_loss,d_loss

    #train each step
    tic=time.time()
    train_model.train()
    conf_losses,paf_losses=np.zeros(shape=(6)),np.zeros(shape=(6))
    avg_conf_loss,avg_paf_loss,avg_total_loss,avg_re_loss=0,0,0,0
    avg_gan_loss,avg_d_loss=0,0
    log('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_steps: {} lr_decay_factor: {} weight_decay_factor: {}'.format(
            n_step, batch_size, lr_init.numpy(), lr_decay_steps, lr_decay_factor, weight_decay_factor))
    for image,gt_label,mask,labeled in train_dataset:
        #extract gt_label
        if(train_model.data_format=="channels_first"):
            gt_conf=gt_label[:,:n_pos,:,:]
            gt_paf=gt_label[:,n_pos:,:,:]
        else:
            gt_conf=gt_label[:,:,:,:n_pos]
            gt_paf=gt_label[:,:,:,n_pos:]
        #learning rate decay
        if(step in lr_decay_steps):
            new_lr_decay = lr_decay_factor**(lr_decay_steps.index(step)+1) 
            lr=lr_init*new_lr_decay

        #optimize one step
        if(domainadapt_flag):
            lambda_d=2/(1+tf.math.exp(-10*(step/n_step)))-1
            pd_conf,pd_paf,total_loss,re_loss,loss_confs,loss_pafs,gan_loss,d_loss=\
                one_step_domainadpat(image.numpy(),gt_conf.numpy(),gt_paf.numpy(),mask.numpy(),labeled.numpy(),train_model,discriminator,lambda_d)
            avg_gan_loss+=gan_loss/log_interval
            avg_d_loss+=d_loss/log_interval
        else:
            pd_conf,pd_paf,total_loss,re_loss,loss_confs,loss_pafs=\
                one_step(image.numpy(),gt_conf.numpy(),gt_paf.numpy(),mask.numpy(),train_model)

        avg_conf_loss+=tf.reduce_mean(loss_confs)/batch_size/log_interval
        avg_paf_loss+=tf.reduce_mean(loss_pafs)/batch_size/log_interval
        avg_total_loss+=total_loss/log_interval
        avg_re_loss+=re_loss/log_interval

        #debug
        for stage_id,(loss_conf,loss_paf) in enumerate(zip(loss_confs,loss_pafs)):
            conf_losses[stage_id]+=loss_conf/batch_size/log_interval
            paf_losses[stage_id]+=loss_paf/batch_size/log_interval

        #save log info periodly
        if((step.numpy()!=0) and (step.numpy()%log_interval)==0):
            tic=time.time()
            log('Train iteration {} / {}: Learning rate {} total_loss:{}, conf_loss:{}, paf_loss:{}, l2_loss {} stage_num:{} time:{}'.format(
                step.numpy(), n_step, lr.numpy(), avg_total_loss, avg_conf_loss, avg_paf_loss, avg_re_loss, len(loss_confs), time.time()-tic))
            for stage_id in range(0,len(loss_confs)):
                log(f"stage_{stage_id} conf_loss:{conf_losses[stage_id]} paf_loss:{paf_losses[stage_id]}")
            if(domainadapt_flag):
                log(f"adaptation loss: g_loss:{avg_gan_loss} d_loss:{avg_d_loss}")
                
            avg_total_loss,avg_conf_loss,avg_paf_loss,avg_re_loss=0,0,0,0
            avg_gan_loss,avg_d_loss=0,0
            conf_losses,paf_losses=np.zeros(shape=(6)),np.zeros(shape=(6))

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
            #save discriminator model
            if(domainadapt_flag):
                dis_save_path=os.path.join(model_dir,"newest_discriminator.npz")
                discriminator.save_weights(dis_save_path)
                log(f"discriminator save_path:{dis_save_path} saved!\n")
            #draw result
            draw_results(image.numpy(), gt_conf.numpy(), pd_conf.numpy(), gt_paf.numpy(), pd_paf.numpy(), mask.numpy(),\
                 vis_dir,'train_%d_' % step,data_format=data_format)

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
    lr_decay_steps = [200000,300000,360000,420000,480000,540000,600000,700000,800000,900000]
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
    parts,limbs,colors=train_model.parts,train_model.limbs,train_model.colors
    data_format=train_model.data_format
    model_dir = config.model.model_dir
    pretrain_model_dir=config.pretrain.pretrain_model_dir
    pretrain_model_path=f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"

    #import kungfu
    from kungfu import current_cluster_size, current_rank
    from kungfu.tensorflow.initializer import broadcast_variables
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, SynchronousAveragingOptimizer, PairAveragingOptimizer
    

    print(f"parallel training using learning rate:{lr_init} batch_size:{batch_size}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    augmentor=Augmentor(hin=hin,win=win,angle_min=-30,angle_max=30,zoom_min=0.5,zoom_max=0.8,flip_list=None)
    preprocessor=PreProcessor(parts=parts,limbs=limbs,hin=hin,win=win,hout=hout,wout=wout,colors=colors,data_format=data_format)
    paramed_map_fn=get_paramed_map_fn(augmentor=augmentor,preprocessor=preprocessor,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096)
    train_dataset = train_dataset.shard(num_shards=current_cluster_size(),index=current_rank())
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)  
    train_dataset = train_dataset.prefetch(64)

    #train model configure 
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    if(config.model.model_type==MODEL.Openpose):
        opt=tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt=tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr)
    ckpt_manager=tf.train.CheckpointManager(ckpt,model_dir,max_to_keep=3)

    #load from ckpt
    try:
        log("loading ckpt...")
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    #load pretrained backbone
    try:
        log("loading pretrained backbone...")
        tl.files.load_and_assign_npz_dict(name=pretrain_model_path,network=train_model.backbone,skip=True)
    except:
        log("pretrained backbone doesn't exist, model backbone are initialized")
    #load model weights
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
    for step_idx,step in enumerate(lr_decay_steps):
        lr_decay_steps[step_idx] = step // current_cluster_size() + 1  # KungFu
    
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
    log('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_steps: {} lr_decay_factor: {}'.format(
            n_step, batch_size, lr_init, lr_decay_steps, lr_decay_factor))
    for image,gt_label,mask in train_dataset:
        #learning rate decay
        if(step in lr_decay_steps):
            new_lr_decay = lr_decay_factor**(float(lr_decay_steps.index(step)+1)) 
            lr=lr_init*new_lr_decay
        #optimize one step
        gt_conf,gt_paf,pd_conf,pd_paf,total_loss,re_loss=one_step(image.numpy(),gt_label.numpy(),mask.numpy(),\
            train_model,step==0)
        #save log info periodly
        if((step.numpy()!=0) and (step.numpy()%log_interval)==0):
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
