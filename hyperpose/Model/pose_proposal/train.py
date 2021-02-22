#!/usr/bin/env python3
import multiprocessing
import os
import time
import matplotlib
matplotlib.use('Agg')
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from functools import partial
from pycocotools.coco import maskUtils
import _pickle as cPickle
from .utils import  draw_results
from .processor import PreProcessor
from ..augmentor import Augmentor
from ..common import log,KUNGFU,get_optim,init_log

def regulize_loss(target_model,weight_decay_factor):
    re_loss=0
    regularizer=tf.keras.regularizers.l2(l=weight_decay_factor)
    for trainable_weight in target_model.trainable_weights:
        re_loss+=regularizer(trainable_weight)
    return re_loss

def _data_aug_fn(image, ground_truth, augmentor, preprocessor, data_format="channels_first"):
    """Data augmentation function."""
    # restore data
    ground_truth = cPickle.loads(ground_truth.numpy())
    image=image.numpy()
    annos = ground_truth["kpt"]
    mask = ground_truth["mask"]
    bbxs = ground_truth["bbx"]
    # kepoint transform
    img_h,img_w,_=image.shape
    annos=np.array(annos).astype(np.float32)
    bbxs=np.array(bbxs).astype(np.float32)
    # decode mask
    h_mask, w_mask, _ = np.shape(image)
    mask_valid = np.ones((h_mask, w_mask), dtype=np.uint8)
    if(mask!=None):
        for seg in mask:
            bin_mask = maskUtils.decode(seg)
            bin_mask = np.logical_not(bin_mask)
            mask_valid = np.bitwise_and(mask_valid, bin_mask)

    # general augmentaton process
    image,annos,mask_valid,bbxs=augmentor.process(image=image,annos=annos,mask_valid=mask_valid,bbxs=bbxs)
    
    # generate result which include proposal region x,y,w,h,edges
    delta,tx,ty,tw,th,te,te_mask=preprocessor.process(annos=annos,mask_valid=mask_valid,bbxs=bbxs)

    #generate output masked image, result map and maskes
    img_mask = mask_valid[:,:,np.newaxis]
    image = image * np.repeat(img_mask, 3, 2)
    if(data_format=="channels_first"):
        image=np.transpose(image,[2,0,1])
    return image,delta,tx,ty,tw,th,te,te_mask

def _map_fn(img_list, annos, data_aug_fn):
    """TF Dataset pipeline."""
    #load data
    image = tf.io.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    #data augmentation using affine transform and get paf maps
    image,delta,tx,ty,tw,th,te,te_mask= tf.py_function(data_aug_fn, [image, annos], [tf.float32, tf.float32,\
         tf.float32,tf.float32, tf.float32, tf.float32,tf.float32, tf.float32])
    #data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=45./255.)   # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image,(delta,tx,ty,tw,th,te,te_mask)

def get_paramed_map_fn(augmentor,preprocessor,data_format="channels_first"):
    paramed_data_aug_fn=partial(_data_aug_fn,augmentor=augmentor,preprocessor=preprocessor,data_format=data_format)
    paramed_map_fn=partial(_map_fn,data_aug_fn=paramed_data_aug_fn)
    return paramed_map_fn

def single_train(train_model,dataset,config):
    '''Single train pipeline of PoseProposal class models

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
    #train hyper paramss
    #dataset params
    n_step = config.train.n_step
    batch_size = config.train.batch_size
    #learning rate params
    lr_init = config.train.lr_init
    lr_decay_factor=config.train.lr_decay_factor
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
    hnei = train_model.hnei
    wnei = train_model.wnei
    parts,limbs,colors=train_model.parts,train_model.limbs,train_model.colors
    data_format=train_model.data_format
    model_dir = config.model.model_dir
    pretrain_model_dir=config.pretrain.pretrain_model_dir
    pretrain_model_path=f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"
    
    log(f"single training using learning rate:{lr_init} batch_size:{batch_size}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    augmentor=Augmentor(hin=hin,win=win,angle_min=-30,angle_max=30,zoom_min=0.5,zoom_max=0.8)
    preprocessor=PreProcessor(parts=parts,limbs=limbs,hin=hin,win=win,hout=hout,wout=wout,hnei=hnei,wnei=wnei,colors=colors,data_format=data_format)
    paramed_map_fn=get_paramed_map_fn(augmentor=augmentor,preprocessor=preprocessor,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    train_dataset = train_dataset.batch(batch_size)  
    train_dataset = train_dataset.prefetch(64)
    
    #train params configure
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    opt=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
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
        log("logging model weights...")
        train_model.load_weights(os.path.join(model_dir,"newest_model.npz"))
    except:
        log("model_path doesn't exist, model parameters are initialized")
        
    #optimize one step
    @tf.function
    def one_step(image,targets,train_model):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            delta,tx,ty,tw,th,te,te_mask=targets
            pc,pi,px,py,pw,ph,pe=train_model.forward(image,is_train=True)
            loss_rsp,loss_iou,loss_coor,loss_size,loss_limb=train_model.cal_loss(delta,tx,ty,tw,th,te,te_mask,pc,pi,px,py,pw,ph,pe)
            pd_loss=loss_rsp+loss_iou+loss_coor+loss_size+loss_limb
            re_loss=regulize_loss(train_model,weight_decay_factor)
            total_loss=pd_loss+re_loss

        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        predicts=(pc,px,py,pw,ph,pe)
        return predicts,targets,pd_loss,re_loss,loss_rsp,loss_iou,loss_coor,loss_size,loss_limb

    #train each step
    tic=time.time()
    train_model.train()
    log(f'Start - n_step: {n_step} batch_size: {batch_size} lr_init: {lr_init} lr_decay_factor: {lr_decay_factor}')
    avg_loss_rsp,avg_loss_iou,avg_loss_coor,avg_loss_size,avg_loss_limb,avg_pd_loss,avg_re_loss=0.,0.,0.,0.,0.,0.,0.
    for image,targets in train_dataset:
        #learning rate decay
        lr=lr_init*(1-step/n_step*lr_decay_factor)
        #optimize one step
        predicts,targets,pd_loss,re_loss,loss_rsp,loss_iou,loss_coor,loss_size,loss_limb\
            =one_step(image,targets,train_model)
        
        avg_loss_rsp+=loss_rsp/log_interval
        avg_loss_iou+=loss_iou/log_interval
        avg_loss_coor+=loss_coor/log_interval
        avg_loss_size+=loss_size/log_interval
        avg_loss_limb+=loss_limb/log_interval
        avg_pd_loss+=pd_loss/log_interval
        avg_re_loss+=re_loss/log_interval
        #save log info periodly
        if((step.numpy()!=0) and (step.numpy()%log_interval)==0):
            tic=time.time()
            log(f"Train iteration {step.numpy()}/{n_step}, learning rate:{lr.numpy()},loss_rsp:{avg_loss_rsp},"+\
                    f"loss_iou:{avg_loss_iou},loss_coor:{avg_loss_coor},loss_size:{avg_loss_size},loss_limb:{avg_loss_limb},"+\
                        f"loss_pd:{avg_pd_loss},loss_re:{avg_re_loss} ,time:{time.time()-tic}")
            avg_loss_rsp,avg_loss_iou,avg_loss_coor,avg_loss_size,avg_loss_limb,avg_pd_loss,avg_re_loss=0.,0.,0.,0.,0.,0.,0.
            
        #save result and ckpt periodly
        if((step.numpy()!=0) and (step.numpy()%save_interval)==0):
            log("saving model ckpt and result...")
            draw_results(image.numpy(),predicts,targets,parts,limbs,save_dir=vis_dir,name=f"ppn_step_{step.numpy()}")
            ckpt_save_path=ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            model_save_path=os.path.join(model_dir,"newest_model.npz")
            train_model.save_weights(model_save_path)
            model_save_path=os.path.join(model_dir,f"step_{step.numpy()}_model.npz")
            train_model.save_weights(model_save_path)
            model_save_npzd_path=os.path.join(model_dir,"newest_model_dict.npz")
            train_model.save_weights(model_save_npzd_path,format="npz_dict")
            log(f"model save_path:{model_save_path} saved!\n")

        #training finished
        if(step==n_step):
            break

def parallel_train(train_model,dataset,config):
    '''Parallel train pipeline of PoseProposal class models

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
    lr_decay_factor=config.train.lr_decay_factor
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
    hnei = train_model.hnei
    wnei = train_model.wnei
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
    augmentor=Augmentor(hin=hin,win=win,angle_min=-30,angle_max=30,zoom_min=0.5,zoom_max=0.8)
    preprocessor=PreProcessor(parts=parts,limbs=limbs,hin=hin,win=win,hout=hout,wout=wout,hneo=hnei,wnei=wnei,colors=colors,data_format=data_format)
    paramed_map_fn=get_paramed_map_fn(augmentor=augmentor,preprocessor=preprocessor,data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096)
    train_dataset = train_dataset.shard(num_shards=current_cluster_size(),index=current_rank())
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=4)
    train_dataset = train_dataset.batch(batch_size)  
    train_dataset = train_dataset.prefetch(buffer_size=2)

    #train model configure
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    opt=tf.keras.optimizers.SGD(learning_rate=lr,momentum=0.9)
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
    
    #Kungfu configure
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

    #optimize one step
    @tf.function
    def one_step(image,targets,train_model,is_first_batch=False):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            delta,tx,ty,tw,th,te,te_mask=targets
            pc,pi,px,py,pw,ph,pe=train_model.forward(image,is_train=True)
            loss_rsp,loss_iou,loss_coor,loss_size,loss_limb=train_model.cal_loss(delta,tx,ty,tw,th,te,te_mask,pc,pi,px,py,pw,ph,pe)
            pd_loss=loss_rsp+loss_iou+loss_coor+loss_size+loss_limb
            re_loss=regulize_loss(train_model,weight_decay_factor)
            total_loss=pd_loss+re_loss

        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        #Kung fu
        if(is_first_batch):
            broadcast_variables(train_model.all_weights)
            broadcast_variables(opt.variables())
        predicts=(pc,px,py,pw,ph,pe)
        return predicts,targets,pd_loss,re_loss,loss_rsp,loss_iou,loss_coor,loss_size,loss_limb

    #train each step
    tic=time.time()
    train_model.train()
    log(f"Worker {current_rank()}: Initialized")
    log(f'Start - n_step: {n_step} batch_size: {batch_size} lr_init: {lr_init} lr_decay_factor: {lr_decay_factor}')
    avg_loss_rsp,avg_loss_iou,avg_loss_coor,avg_loss_size,avg_loss_limb,avg_pd_loss,avg_re_loss=0.,0.,0.,0.,0.,0.,0.
    for image,targets in train_dataset:
        #learning rate decay
        lr=lr_init*(1-step/n_step*lr_decay_factor)
        #optimize one step
        predicts,targets,pd_loss,re_loss,loss_rsp,loss_iou,loss_coor,loss_size,loss_limb=one_step(image,targets,train_model)
        
        avg_loss_rsp+=loss_rsp/log_interval
        avg_loss_iou+=loss_iou/log_interval
        avg_loss_coor+=loss_coor/log_interval
        avg_loss_size+=loss_size/log_interval
        avg_loss_limb+=loss_limb/log_interval
        avg_pd_loss+=pd_loss/log_interval
        avg_re_loss+=re_loss/log_interval
        
        #save log info periodly
        if((step.numpy()!=0) and (step.numpy()%log_interval)==0):
            tic=time.time()
            log(f"worker:{current_rank()} Train iteration {step.numpy()}/{n_step}, learning rate:{lr.numpy()},"+\
                    f"loss_rsp:{avg_loss_rsp},loss_iou:{avg_loss_iou},loss_coor:{avg_loss_coor},loss_size:{avg_loss_size},"+\
                        f"loss_limb:{avg_loss_limb},loss_pd:{avg_pd_loss},loss_re:{avg_re_loss} ,time:{time.time()-tic}")
            avg_loss_rsp,avg_loss_iou,avg_loss_coor,avg_loss_size,avg_loss_limb,avg_pd_loss,avg_re_loss=0.,0.,0.,0.,0.,0.,0.

        #save result and ckpt periodly
        if((step!=0) and (step%save_interval)==0):
            log("saving model ckpt and result...")
            draw_results(image.numpy(),predicts,targets,parts,limbs,save_dir=vis_dir,name=f"ppn_step_{step.numpy()}")
            ckpt_save_path=ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            model_save_path=os.path.join(model_dir,"newest_model.npz")
            train_model.save_weights(model_save_path)
            log(f"model save_path:{model_save_path} saved!\n")

        #training finished
        if(step==n_step):
            break

