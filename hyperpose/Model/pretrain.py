import os
import cv2
import numpy as np
import multiprocessing
import tensorflow as tf
import tensorlayer as tl
from functools import partial
from .common import regulize_loss
from .common import log_train as log

def _data_aug(image,hin,win,data_format):
    image=image.numpy()
    image=cv2.resize(image,(hin,win))
    if(data_format=="channels_first"):
        image=np.transpose(image,[2,0,1])
    return image

def train_map_fn(image_path,image_label,data_aug):
    """TF Dataset pipeline."""
    #load data
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.py_function(data_aug, [image], [tf.float32])[0]
    #data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=45./255.)   # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image, image_label

def val_map_fn(image_path,image_label,data_aug):
    #load data
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.py_function(data_aug, [image], [tf.float32])[0]
    return image,image_label

def single_pretrain(model,dataset,config):
    init_log(config)
    lr_init=config.pretrain.lr_init
    lr_decay_step=config.pretrain.lr_decay_step
    batch_size=config.pretrain.batch_size
    total_step=config.pretrain.total_step
    log_interval=config.pretrain.log_interval
    val_interval=config.pretrain.val_interval
    save_interval=config.pretrain.save_interval
    pretrain_model_dir=config.pretrain.pretrain_model_dir
    weight_decay_factor=config.pretrain.weight_decay_factor

    print(f"starting to pretrain model backbone with learning rate:{lr_init} batch_size:{batch_size}")
    print(f"pretraining model_type:{config.model.model_type} model_backbone:{model.backbone.name}")
    #training dataset configure with shuffle,augmentation,and prefetch
    train_dataset=dataset.get_train_dataset()
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    data_aug=partial(_data_aug,hin=224,win=224,data_format=model.data_format)
    train_dataset = train_dataset.map(partial(train_map_fn,data_aug=data_aug),num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    train_dataset = train_dataset.batch(batch_size)  
    train_dataset = train_dataset.prefetch(8)
    
    #train configure
    step=tf.Variable(1, trainable=False)
    lr=tf.Variable(lr_init,trainable=False)
    opt=tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt=tf.train.Checkpoint(step=step,optimizer=opt,lr=lr)
    ckpt_manager=tf.train.CheckpointManager(ckpt,pretrain_model_dir,max_to_keep=3)

    train_model=model.backbone
    train_model.train()
    #load from ckpt
    try:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, learning rate, step and optimizer are initialized")
    try:
        train_model.load_weights(os.path.join(pretrain_model_dir,f"newest_{train_model.name}.npz"),format="npz_dict")
    except:
        log("model_path doesn't exist, model parameters are initialized")
    
    total_pd_loss,total_re_loss,total_top1_acc_num,total_top5_acc_num,total_img_num=0,0,0,0,0
    max_eval_acc=0
    stuck_time=0

    #optimize one step
    @tf.function
    def one_step(image,label,train_model):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            predict=train_model.forward(image)
            pd_loss=train_model.cal_loss(label,predict)
            re_loss=regulize_loss(train_model,weight_decay_factor)
            total_loss=pd_loss+re_loss

        top1_acc_num=tf.reduce_sum(tf.where(tf.math.in_top_k(label,predict,1),1,0))
        top5_acc_num=tf.reduce_sum(tf.where(tf.math.in_top_k(label,predict,5),1,0))
        gradients=tape.gradient(total_loss,train_model.trainable_weights)
        opt.apply_gradients(zip(gradients,train_model.trainable_weights))
        return top1_acc_num,top5_acc_num,pd_loss,re_loss,predict
    
    for image,label in train_dataset:
        top1_acc_num,top5_acc_num,pd_loss,re_loss,predict=one_step(image.numpy(),label.numpy(),train_model)
        total_pd_loss+=pd_loss/log_interval
        total_re_loss+=re_loss/log_interval
        total_top1_acc_num+=top1_acc_num
        total_top5_acc_num+=top5_acc_num
        total_img_num+=batch_size

        if(step%lr_decay_step==0):
            lr=lr/5

        #log info
        if(step!=0 and step%log_interval==0):
            print("Train iteration {} / {}: Learning rate:{} total_loss:{} pd_loss:{} re_loss:{} accuracy_top1:{} accuracy_top5:{}".format(
                step.numpy(),total_step, lr.numpy(), total_pd_loss+total_re_loss, total_pd_loss, total_re_loss, total_top1_acc_num/total_img_num, total_top5_acc_num/total_img_num))
            total_pd_loss,total_re_loss=0.0,0.0
            total_top1_acc_num,total_top5_acc_num,total_img_num=0,0,0

        #save model
        if(step!=0 and step%save_interval==0):
            ckpt_save_path=ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            model_save_path=os.path.join(pretrain_model_dir,f"newest_{train_model.name}.npz")
            train_model.save_weights(model_save_path,format="npz_dict")
            log(f"model save_path:{model_save_path} saved!\n")
        
        #validate model
        if(step!=0 and step%val_interval==0):
            train_model.eval()
            eval_acc=single_val(train_model,dataset,config)
            print(f"current validate: eval_acc:{eval_acc} max_eval_acc:{max_eval_acc} stuck_time:{stuck_time}")
            if(eval_acc<max_eval_acc):
                stuck_time+=1
            else:
                max_eval_acc=eval_acc
            if(stuck_time>=3):
                lr=lr/5
                stuck_time=0
            train_model.train()

        if(step==total_step):
            break
        

def single_val(val_model,dataset,config):
    total_val_num=config.pretrain.val_num
    log("starting validate... ")
    val_dataset=dataset.get_eval_dataset()
    val_dataset = val_dataset.shuffle(buffer_size=4096)
    data_aug=partial(_data_aug,hin=224,win=224,data_format=val_model.data_format)
    val_dataset = val_dataset.map(partial(val_map_fn,data_aug=data_aug),num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    val_dataset = val_dataset.batch(64)  
    val_dataset = val_dataset.prefetch(64)

    total_top1_acc_num,total_top5_acc_num,total_img_num=0,0,0
    @tf.function
    def one_step(image,label,val_model):
        predict=val_model.forward(image)
        val_top1_acc_num=tf.reduce_sum(tf.where(tf.math.in_top_k(label,predict,1),1,0))
        val_top5_acc_num=tf.reduce_sum(tf.where(tf.math.in_top_k(label,predict,5),1,0))
        return val_top1_acc_num,val_top5_acc_num
    
    for image,label in val_dataset:
        top1_acc_num,top5_acc_num=one_step(image.numpy(),label.numpy(),val_model)
        total_top1_acc_num+=top1_acc_num
        total_top5_acc_num+=top5_acc_num
        total_img_num+=image.shape[0]
        if(total_img_num>=total_val_num):
            break
    print(f"validation accuracy_top1:{total_top1_acc_num/total_img_num} accuracy_top5:{total_top5_acc_num/total_img_num}")
    return total_top1_acc_num/total_img_num


