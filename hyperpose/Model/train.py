#!/usr/bin/env python3
import os
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import tensorlayer as tl
import _pickle as cPickle
from functools import partial, reduce
from .common import KUNGFU
from .common import log_train as log
from .domainadapt import Discriminator
from .common import decode_mask,get_num_parallel_calls
from .metrics import MetricManager
from .augmentor import BasicAugmentor
from .processor import BasicPreProcessor
from .processor import BasicPostProcessor
from .processor import BasicVisualizer
from .common import to_tensor_dict


def _data_aug_fn(image, ground_truth, augmentor:BasicAugmentor, preprocessor:BasicPreProcessor, data_format="channels_first"):
    """Data augmentation function."""
    # restore data
    image = image.numpy()
    ground_truth = cPickle.loads(ground_truth.numpy())
    annos = ground_truth["kpt"]
    meta_mask = ground_truth["mask"]
    bbxs = ground_truth["bbxs"]

    # decode mask
    mask = decode_mask(meta_mask)
    if(mask is None):
        mask = np.ones_like(image)[:,:,0].astype(np.uint8)

    # general augmentaton process
    image, annos, mask, bbxs = augmentor.process(image=image, annos=annos, mask=mask, bbxs=bbxs)
    mask = mask[:,:,np.newaxis]
    image = image * mask

    # TODO: all process are in channels_first format
    image = np.transpose(image, [2, 0, 1])
    mask = np.transpose(mask, [2, 0, 1])

    # generate result including heatmap and vectormap
    target_x = preprocessor.process(annos=annos, mask=mask, bbxs=bbxs)
    target_x = cPickle.dumps(target_x)

    return image, mask, target_x


def _map_fn(img_list, annos, data_aug_fn):
    """TF Dataset pipeline."""

    # load data
    image = tf.io.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # data augmentation using affine transform and get paf maps
    image, mask, target_x= tf.py_function(data_aug_fn, [image, annos],
                                                    [tf.float32, tf.float32, tf.string])

    # data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=35. / 255.)  # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image, mask, target_x

def get_paramed_map_fn(augmentor, preprocessor, data_format="channels_first"):
    paramed_data_aug_fn = partial(_data_aug_fn, augmentor=augmentor, preprocessor=preprocessor, data_format=data_format)
    paramed_map_fn = partial(_map_fn, data_aug_fn=paramed_data_aug_fn)
    return paramed_map_fn

def _dmadapt_data_aug_fn(image, augmentor, data_format="channels_first"):
    image = image.numpy()
    image = augmentor.process_only_image(image)
    if(data_format=="channels_first"):
        image = np.transpose(image, [2,0,1])
    return image

def _dmadapt_map_fn(image, aug_fn):
    image = tf.py_function(aug_fn, [image], tf.float32)
    return image

def get_paramed_dmadapt_map_fn(augmentor):
    paramed_dmadapt_aug_fn = partial(_dmadapt_data_aug_fn, augmentor=augmentor)
    paramed_dmadpat_map_fn = partial(_dmadapt_map_fn, aug_fn=paramed_dmadapt_aug_fn)
    return paramed_dmadpat_map_fn


def single_train(train_model, dataset, config, augmentor:BasicAugmentor, \
                    preprocessor:BasicPreProcessor,postprocessor:BasicPostProcessor,visualizer:BasicVisualizer):
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

    # train hyper params
    # dataset params
    total_step = config.train.n_step
    batch_size = config.train.batch_size
    # learning rate params
    lr_init = config.train.lr_init
    lr_decay_factor = config.train.lr_decay_factor
    lr_decay_steps = [200000, 300000, 360000, 420000, 480000, 540000, 600000, 700000, 800000, 900000]
    weight_decay_factor = config.train.weight_decay_factor
    # log and checkpoint params
    log_interval = config.log.log_interval
    vis_interval =  config.train.vis_interval
    save_interval = config.train.save_interval

    # model hyper params
    data_format = train_model.data_format
    model_dir = config.model.model_dir
    pretrain_model_dir = config.pretrain.pretrain_model_dir
    pretrain_model_path = f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"

    # metrics
    metric_manager = MetricManager()

    # initializing train dataset
    train_dataset = dataset.get_train_dataset()
    epoch_size = dataset.get_train_datasize()//batch_size
    paramed_map_fn = get_paramed_map_fn(augmentor=augmentor, preprocessor=preprocessor, data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=get_num_parallel_calls())
    train_dataset = train_dataset.batch(config.train.batch_size)
    train_dataset = train_dataset.prefetch(3)
    train_dataset_iter = iter(train_dataset)

    #train configure
    save_step = tf.Variable(1, trainable=False)
    save_lr = tf.Variable(lr_init, trainable=False)
    opt = tf.keras.optimizers.Adam(learning_rate=save_lr)
    domainadapt_flag = config.data.domainadapt_flag
    total_epoch = total_step//epoch_size

    #domain adaptation params
    if (not domainadapt_flag):
        ckpt = tf.train.Checkpoint(save_step=save_step, save_lr=save_lr, opt=opt)
    else:
        log("Domain adaptaion in training enabled!")
        # weight param
        lambda_adapt=1e-4
        # construct discrminator model
        feature_hin = train_model.hin//train_model.backbone.scale_size
        feature_win = train_model.win//train_model.backbone.scale_size
        in_channels = train_model.backbone.out_channels
        adapt_dis = Discriminator(feature_hin, feature_win, in_channels, data_format=data_format)
        opt_d = tf.keras.optimizers.Adam(learning_rate=save_lr)
        ckpt = tf.train.Checkpoint(save_step=save_step, save_lr=save_lr, opt=opt, opt_d=opt_d)
        # construct domain adaptation dataset
        dmadapt_train_dataset = dataset.get_dmadapt_train_dataset()
        paramed_dmadapt_map_fn = get_paramed_dmadapt_map_fn(augmentor)
        dmadapt_train_dataset = dmadapt_train_dataset.map(paramed_dmadapt_map_fn, num_parallel_calls=get_num_parallel_calls())
        dmadapt_train_dataset = dmadapt_train_dataset.shuffle(buffer_size=4096).repeat()
        dmadapt_train_dataset = dmadapt_train_dataset.batch(config.train.batch_size)
        dmadapt_train_dataset = dmadapt_train_dataset.prefetch(3)
        dmadapt_train_dataset_iter = iter(dmadapt_train_dataset)


    #load from ckpt
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    try:
        log("loading ckpt...")
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    #load pretrained backbone
    try:
        log("loading pretrained backbone...")
        train_model.backbone.load_weight(pretrain_model_path, format="npz_dict")
    except:
        log("pretrained backbone doesn't exist, model backbone are initialized")
    #load model weights
    try:
        log("loading saved training model weights...")
        train_model.load_weights(os.path.join(model_dir, "newest_model.npz"), format="npz_dict")
    except:
        log("model_path doesn't exist, model parameters are initialized")
    if (domainadapt_flag):
        try:
            log("loading saved domain adaptation discriminator weight...")
            adapt_dis.load_weights(os.path.join(model_dir, "newest_discriminator.npz"))
        except:
            log("discriminator path doesn't exist, discriminator parameters are initialized")

    
    log(f"single training using learning rate:{lr_init} batch_size:{batch_size}")
    step = save_step.numpy()
    lr = save_lr.numpy()

    for lr_decay_step in lr_decay_steps:
        if (step > lr_decay_step):
            lr = lr * lr_decay_factor

    # optimize one step
    def optimize_step(image, mask, target_x, train_model, metric_manager: MetricManager):
        # tape
        with tf.GradientTape() as tape:
            predict_x = train_model.forward(x=image, is_train=True, ret_backbone=domainadapt_flag)
            total_loss = train_model.cal_loss(predict_x=predict_x, target_x=target_x, \
                                                        mask=mask, metric_manager=metric_manager)
        # optimize model
        gradients = tape.gradient(total_loss, train_model.trainable_weights)
        opt.apply_gradients(zip(gradients, train_model.trainable_weights))
        return predict_x
    
    def optimize_step_dmadapt(image_src, image_dst, train_model, adapt_dis: Discriminator, metric_manager: MetricManager):
        # tape
        with tf.GradientTape(persistent=True) as tape:
            # feature extraction
            # src feature
            predict_src = train_model.forward(x=image_src, is_train=True, ret_backbone=True)
            backbone_feature_src = predict_src["backbone_features"]
            adapt_pd_src = adapt_dis.forward(backbone_feature_src)
            # dst feature
            predict_dst = train_model.forward(x=image_dst, is_train=True, ret_backbone=True)
            backbone_feature_dst =  predict_dst["backbone_features"]
            adapt_pd_dst = adapt_dis.forward(backbone_feature_dst)

            # loss calculation
            # loss of g
            g_adapt_loss = adapt_dis.cal_loss(x=adapt_pd_dst, label=True)*lambda_adapt
            # loss of d 
            d_adapt_loss_src = adapt_dis.cal_loss(x=adapt_pd_src, label=True)
            d_adapt_loss_dst = adapt_dis.cal_loss(x=adapt_pd_dst, label=False)
            d_adapt_loss = (d_adapt_loss_src+d_adapt_loss_dst)/2

        # optimize model
        g_gradient = tape.gradient(g_adapt_loss, train_model.trainable_weights)
        opt.apply_gradients(zip(g_gradient, train_model.trainable_weights))
        metric_manager.update("model/g_adapt_loss",g_adapt_loss)
        # optimize dis
        d_gradients = tape.gradient(d_adapt_loss, adapt_dis.trainable_weights)
        opt_d.apply_gradients(zip(d_gradients, adapt_dis.trainable_weights))
        metric_manager.update("dis/d_adapt_loss_src",d_adapt_loss_src)
        metric_manager.update("dis/d_adapt_loss_dst",d_adapt_loss_dst)
        # delete persistent tape
        del tape
        return predict_dst

    # formal training procedure
    train_model.train()
    cur_epoch = step // epoch_size +1
    log(f"Start Training- total_epoch: {total_epoch} total_step: {total_step} current_epoch:{cur_epoch} "\
        +f"current_step:{step} batch_size:{batch_size} lr_init:{lr_init} lr_decay_steps:{lr_decay_steps} "\
        +f"lr_decay_factor:{lr_decay_factor} weight_decay_factor:{weight_decay_factor}" )
    for epoch_idx in range(cur_epoch,total_epoch):
        log(f"Epoch {epoch_idx}/{total_epoch}:")
        for _ in tqdm(range(0,epoch_size)):
            step+=1
            metric_manager.start_timing()
            image, mask, target_list = next(train_dataset_iter)
            # extract gt_label
            target_list = [cPickle.loads(target) for target in target_list.numpy()]
            target_x = {key:[] for key,value in target_list[0].items()}
            target_x = reduce(lambda x, y: {key:x[key]+[y[key]] for key,value in x.items()},[target_x]+target_list)
            target_x = {key:np.stack(value) for key,value in target_x.items()}
            target_x = to_tensor_dict(target_x)

            # learning rate decay
            if (step in lr_decay_steps):
                new_lr_decay = lr_decay_factor**(lr_decay_steps.index(step) + 1)
                lr = lr_init * new_lr_decay

            # optimize one step
            predict_x = optimize_step(image, mask, target_x, train_model, metric_manager)

            # optimize domain adaptation
            if(domainadapt_flag):
                src_image = image
                dst_image = next(dmadapt_train_dataset_iter)
                predict_dst = optimize_step_dmadapt(src_image, dst_image, train_model, adapt_dis, metric_manager)

            # log info periodly
            if ((step != 0) and (step % log_interval) == 0):
                log(f"Train Epoch={epoch_idx} / {total_epoch}, Step={step} / {total_step}: learning_rate: {lr:.6e} {metric_manager.report_timing()}\n"\
                        +f"{metric_manager.report_train()} ")

            # visualize periodly
            if ((step != 0) and (step % vis_interval) == 0):
                log(f"Visualizing prediction maps and target maps")
                predict_x = train_model.forward(x=image, is_train=False)
                visualizer.visualize_compare(image_batch=image.numpy(), mask_batch=mask.numpy(), predict_x=predict_x, target_x=target_x,\
                                                    name=f"train_{step}")

            # save result and ckpt periodly
            if ((step!= 0) and (step % save_interval) == 0):
                # save ckpt
                log("saving model ckpt and result...")
                save_step.assign(step)
                save_lr.assign(lr)
                ckpt_save_path = ckpt_manager.save()
                log(f"ckpt save_path:{ckpt_save_path} saved!\n")
                # save train model
                model_save_path = os.path.join(model_dir, "newest_model.npz")
                train_model.save_weights(model_save_path, format="npz_dict")
                log(f"model save_path:{model_save_path} saved!\n")
                # save discriminator model
                if (domainadapt_flag):
                    dis_save_path = os.path.join(model_dir, "newest_discriminator.npz")
                    adapt_dis.save_weights(dis_save_path, format="npz_dict")
                    log(f"discriminator save_path:{dis_save_path} saved!\n")

def parallel_train(train_model, dataset, config, augmentor:BasicAugmentor, \
                        preprocessor:BasicPreProcessor,postprocessor:BasicPostProcessor,visualizer=BasicVisualizer):
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

    # train hyper params
    # dataset params
    total_step = config.train.n_step
    batch_size = config.train.batch_size
    # learning rate params
    lr_init = config.train.lr_init
    lr_decay_factor = config.train.lr_decay_factor
    lr_decay_steps = [200000, 300000, 360000, 420000, 480000, 540000, 600000, 700000, 800000, 900000]
    weight_decay_factor = config.train.weight_decay_factor
    # log and checkpoint params
    log_interval = config.log.log_interval
    vis_interval =  config.train.vis_interval
    save_interval = config.train.save_interval
    vis_dir = config.train.vis_dir

    # model hyper params
    hin = train_model.hin
    win = train_model.win
    hout = train_model.hout
    wout = train_model.wout
    parts, limbs, colors = train_model.parts, train_model.limbs, train_model.colors
    data_format = train_model.data_format
    model_dir = config.model.model_dir
    pretrain_model_dir = config.pretrain.pretrain_model_dir
    pretrain_model_path = f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"
    
    # metrics
    metric_manager = MetricManager()

    # initializing train dataset
    train_dataset = dataset.get_train_dataset()
    epoch_size = dataset.get_train_datasize()//batch_size
    paramed_map_fn = get_paramed_map_fn(augmentor=augmentor, preprocessor=preprocessor, data_format=data_format)
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=get_num_parallel_calls())
    train_dataset = train_dataset.batch(config.train.batch_size)
    train_dataset = train_dataset.prefetch(3)
    train_dataset_iter = iter(train_dataset)

    #train configure
    save_step = tf.Variable(1, trainable=False)
    save_lr = tf.Variable(lr_init, trainable=False)
    opt = tf.keras.optimizers.Adam(learning_rate=save_lr)
    domainadapt_flag = config.data.domainadapt_flag
    total_epoch = total_step//epoch_size

    #domain adaptation params
    if (not domainadapt_flag):
        ckpt = tf.train.Checkpoint(save_step=save_step, save_lr=save_lr, opt=opt)
    else:
        log("Domain adaptaion in training enabled!")
        # weight param
        lambda_adapt=1e-4
        # construct discrminator model
        feature_hin = train_model.hin//train_model.backbone.scale_size
        feature_win = train_model.win//train_model.backbone.scale_size
        in_channels = train_model.backbone.out_channels
        adapt_dis = Discriminator(feature_hin, feature_win, in_channels, data_format=data_format)
        opt_d = tf.keras.optimizers.Adam(learning_rate=save_lr)
        ckpt = tf.train.Checkpoint(save_step=save_step, save_lr=save_lr, opt=opt, opt_d=opt_d)
        # construct domain adaptation dataset
        dmadapt_train_dataset = dataset.get_dmadapt_train_dataset()
        paramed_dmadapt_map_fn = get_paramed_dmadapt_map_fn(augmentor)
        dmadapt_train_dataset = dmadapt_train_dataset.map(paramed_dmadapt_map_fn, num_parallel_calls=get_num_parallel_calls())
        dmadapt_train_dataset = dmadapt_train_dataset.shuffle(buffer_size=4096).repeat()
        dmadapt_train_dataset = dmadapt_train_dataset.batch(config.train.batch_size)
        dmadapt_train_dataset = dmadapt_train_dataset.prefetch(3)
        dmadapt_train_dataset_iter = iter(dmadapt_train_dataset)


    #load from ckpt
    ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
    try:
        log("loading ckpt...")
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, step and optimizer are initialized")
    #load pretrained backbone
    try:
        log("loading pretrained backbone...")
        tl.files.load_and_assign_npz_dict(name=pretrain_model_path, network=train_model.backbone, skip=True)
    except:
        log("pretrained backbone doesn't exist, model backbone are initialized")
    #load model weights
    try:
        log("loading saved training model weights...")
        train_model.load_weights(os.path.join(model_dir, "newest_model.npz"))
    except:
        log("model_path doesn't exist, model parameters are initialized")
    if (domainadapt_flag):
        try:
            log("loading saved domain adaptation discriminator weight...")
            adapt_dis.load_weights(os.path.join(model_dir, "newest_discriminator.npz"))
        except:
            log("discriminator path doesn't exist, discriminator parameters are initialized")

    
    log(f"Parallel training using learning rate:{lr_init} batch_size:{batch_size}")
    step = save_step.numpy()
    lr = save_lr.numpy()

    #import kungfu
    from kungfu.python import current_cluster_size, current_rank
    from kungfu.tensorflow.initializer import broadcast_variables
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, SynchronousAveragingOptimizer, PairAveragingOptimizer

    total_step = total_step // current_cluster_size() + 1  # KungFu
    total_epoch = total_epoch // current_cluster_size() +1 # KungFu
    for step_idx, decay_step in enumerate(lr_decay_steps):
        lr_decay_steps[step_idx] = decay_step // current_cluster_size() + 1  # KungFu

    # optimize one step
    def optimize_step(image, mask, target_x, train_model, metric_manager: MetricManager):
        # tape
        with tf.GradientTape() as tape:
            predict_x = train_model.forward(x=image, is_train=True, ret_backbone=domainadapt_flag)
            total_loss = train_model.cal_loss(predict_x=predict_x, target_x=target_x, \
                                                        mask=mask, metric_manager=metric_manager)
        # optimize model
        gradients = tape.gradient(total_loss, train_model.trainable_weights)
        opt.apply_gradients(zip(gradients, train_model.trainable_weights))
        return predict_x
    
    def optimize_step_dmadapt(image_src, image_dst, train_model, adapt_dis: Discriminator, metric_manager: MetricManager):
        # tape
        with tf.GradientTape(persistent=True) as tape:
            # feature extraction
            # src feature
            predict_src = train_model.forward(x=image_src, is_train=True, ret_backbone=True)
            backbone_feature_src = predict_src["backbone_features"]
            adapt_pd_src = adapt_dis.forward(backbone_feature_src)
            # dst feature
            predict_dst = train_model.forward(x=image_dst, is_train=True, ret_backbone=True)
            backbone_feature_dst =  predict_dst["backbone_features"]
            adapt_pd_dst = adapt_dis.forward(backbone_feature_dst)

            # loss calculation
            # loss of g
            g_adapt_loss = adapt_dis.cal_loss(x=adapt_pd_dst, label=True)*lambda_adapt
            # loss of d 
            d_adapt_loss_src = adapt_dis.cal_loss(x=adapt_pd_src, label=True)
            d_adapt_loss_dst = adapt_dis.cal_loss(x=adapt_pd_dst, label=False)
            d_adapt_loss = (d_adapt_loss_src+d_adapt_loss_dst)/2

        # optimize model
        g_gradient = tape.gradient(g_adapt_loss, train_model.trainable_weights)
        opt.apply_gradients(zip(g_gradient, train_model.trainable_weights))
        metric_manager.update("model/g_adapt_loss",g_adapt_loss)
        # optimize dis
        d_gradients = tape.gradient(d_adapt_loss, adapt_dis.trainable_weights)
        opt_d.apply_gradients(zip(d_gradients, adapt_dis.trainable_weights))
        metric_manager.update("dis/d_adapt_loss_src",d_adapt_loss_src)
        metric_manager.update("dis/d_adapt_loss_dst",d_adapt_loss_dst)
        # delete persistent tape
        del tape
        return predict_dst

    # formal training procedure
    

    # KungFu configure
    kungfu_option = config.train.kungfu_option
    if kungfu_option == KUNGFU.Sync_sgd:
        print("using Kungfu.SynchronousSGDOptimizer!")
        opt = SynchronousSGDOptimizer(opt)
    elif kungfu_option == KUNGFU.Sync_avg:
        print("using Kungfu.SynchronousAveragingOptimize!")
        opt = SynchronousAveragingOptimizer(opt)
    elif kungfu_option == KUNGFU.Pair_avg:
        print("using Kungfu.PairAveragingOptimizer!")
        opt = PairAveragingOptimizer(opt)

    train_model.train()
    cur_epoch = step // epoch_size +1
    log(f"Start Training- total_epoch: {total_epoch} total_step: {total_step} current_epoch:{cur_epoch} "\
        +f"current_step:{step} batch_size:{batch_size} lr_init:{lr_init} lr_decay_steps:{lr_decay_steps} "\
        +f"lr_decay_factor:{lr_decay_factor} weight_decay_factor:{weight_decay_factor}" )
    for epoch_idx in range(cur_epoch,total_epoch):
        log(f"Epoch {epoch_idx}/{total_epoch}:")
        for _ in tqdm(range(0,epoch_size)):
            step+=1
            metric_manager.start_timing()
            image, mask, target_list = next(train_dataset_iter)
            # extract gt_label
            target_list = [cPickle.loads(target) for target in target_list.numpy()]
            target_x = {key:[] for key,value in target_list[0].items()}
            target_x = reduce(lambda x, y: {key:x[key]+[y[key]] for key,value in x.items()},[target_x]+target_list)
            target_x = {key:np.stack(value) for key,value in target_x.items()}            
            target_x = to_tensor_dict(target_x)


            # learning rate decay
            if (step in lr_decay_steps):
                new_lr_decay = lr_decay_factor**(lr_decay_steps.index(step) + 1)
                lr = lr_init * new_lr_decay

            # optimize one step
            predict_x = optimize_step(image, mask, target_x, train_model, metric_manager)

            # optimize domain adaptation
            if(domainadapt_flag):
                src_image = image
                dst_image = next(dmadapt_train_dataset_iter)
                predict_dst = optimize_step_dmadapt(src_image, dst_image, train_model, adapt_dis, metric_manager)
            
            if(step==1):
                broadcast_variables(train_model.all_weights)
                broadcast_variables(opt.variables())

            # log info periodly
            if ((step != 0) and (step % log_interval) == 0):
                log(f"Train Epoch={epoch_idx} / {total_epoch}, Step={step} / {total_step}: learning_rate: {lr:.6e} {metric_manager.report_timing()}\n"\
                        +f"{metric_manager.report_train()} ")

            # visualize periodly
            if ((step != 0) and (step % vis_interval) == 0 and current_rank() == 0):
                log(f"Visualizing prediction maps and target maps")
                visualizer.visual_compare(image_batch=image.numpy(), mask_batch=mask.numpy(), predict_x=predict_x, target_x=target_x,\
                                                    name=f"train_{step}")

            # save result and ckpt periodly
            if ((step!= 0) and (step % save_interval) == 0 and current_rank() == 0):
                # save ckpt
                log("saving model ckpt and result...")
                save_step.assign(step)
                save_lr.assign(lr)
                ckpt_save_path = ckpt_manager.save()
                log(f"ckpt save_path:{ckpt_save_path} saved!\n")
                # save train model
                model_save_path = os.path.join(model_dir, "newest_model.npz")
                train_model.save_weights(model_save_path)
                log(f"model save_path:{model_save_path} saved!\n")
                # save discriminator model
                if (domainadapt_flag):
                    dis_save_path = os.path.join(model_dir, "newest_discriminator.npz")
                    adapt_dis.save_weights(dis_save_path)
                    log(f"discriminator save_path:{dis_save_path} saved!\n")