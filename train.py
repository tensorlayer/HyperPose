#!/usr/bin/env python3

import math
import multiprocessing
import os
import time
import sys
import argparse

import cv2
import matplotlib
matplotlib.use('Agg')

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from pycocotools.coco import maskUtils

import _pickle as cPickle

sys.path.append('.')

from train_config import config
from openpose_plus.models import model
from openpose_plus.utils import PoseInfo, draw_results, get_heatmap, get_vectormap, load_mscoco_dataset, tf_repeat

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

tl.files.exists_or_mkdir(config.LOG.vis_path, verbose=False)  # to save visualization results
tl.files.exists_or_mkdir(config.MODEL.model_path, verbose=False)  # to save model files

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# FIXME: Don't use global variables.
# define hyper-parameters for training
batch_size = config.TRAIN.batch_size
lr_decay_every_step = config.TRAIN.lr_decay_every_step
n_step = config.TRAIN.n_step
save_interval = config.TRAIN.save_interval
weight_decay_factor = config.TRAIN.weight_decay_factor
lr_init = config.TRAIN.lr_init
lr_decay_factor = config.TRAIN.lr_decay_factor

# FIXME: Don't use global variables.
# define hyper-parameters for model
model_path = config.MODEL.model_path
n_pos = config.MODEL.n_pos
hin = config.MODEL.hin
win = config.MODEL.win
hout = config.MODEL.hout
wout = config.MODEL.wout


def get_pose_data_list(im_path, ann_path):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    print("[x] Get pose data from {}".format(im_path))
    data = PoseInfo(im_path, ann_path, False)
    imgs_file_list = data.get_image_list()
    objs_info_list = data.get_joint_list()
    mask_list = data.get_mask()
    targets = list(zip(objs_info_list, mask_list))
    if len(imgs_file_list) != len(objs_info_list):
        raise Exception("number of images and annotations do not match")
    else:
        print("{} has {} images".format(im_path, len(imgs_file_list)))
    return imgs_file_list, objs_info_list, mask_list, targets


def make_model(img, results, mask, is_train=True, reuse=False):
    confs = results[:, :, :, :n_pos]
    pafs = results[:, :, :, n_pos:]
    m1 = tf_repeat(mask, [1, 1, 1, n_pos])
    m2 = tf_repeat(mask, [1, 1, 1, n_pos * 2])

    cnn, b1_list, b2_list, net = model(img, n_pos, m1, m2, is_train, reuse)

    # define loss
    losses = []
    last_losses_l1 = []
    last_losses_l2 = []
    stage_losses = []

    for idx, (l1, l2) in enumerate(zip(b1_list, b2_list)):
        loss_l1 = tf.nn.l2_loss((l1.outputs - confs) * m1)
        loss_l2 = tf.nn.l2_loss((l2.outputs - pafs) * m2)

        losses.append(tf.reduce_mean([loss_l1, loss_l2]))
        stage_losses.append(loss_l1 / batch_size)
        stage_losses.append(loss_l2 / batch_size)

    last_conf = b1_list[-1].outputs
    last_paf = b2_list[-1].outputs
    last_losses_l1.append(loss_l1)
    last_losses_l2.append(loss_l2)
    l2_loss = 0.0

    for p in tl.layers.get_variables_with_name('kernel', True, True):
        l2_loss += tf.contrib.layers.l2_regularizer(weight_decay_factor)(p)
    total_loss = tf.reduce_sum(losses) / batch_size + l2_loss

    log_tensors = {'total_loss': total_loss, 'stage_losses': stage_losses, 'l2_loss': l2_loss}
    net.cnn = cnn
    net.img = img  # net input
    net.last_conf = last_conf  # net output
    net.last_paf = last_paf  # net output
    net.confs = confs  # GT
    net.pafs = pafs  # GT
    net.m1 = m1  # mask1, GT
    net.m2 = m2  # mask2, GT
    net.stage_losses = stage_losses
    net.l2_loss = l2_loss
    return net, total_loss, log_tensors
    # return total_loss, last_conf, stage_losses, l2_loss, cnn, last_paf, img, confs, pafs, m1, net


def _data_aug_fn(image, ground_truth):
    """Data augmentation function."""
    ground_truth = cPickle.loads(ground_truth)
    ground_truth = list(ground_truth)

    annos = ground_truth[0]
    mask = ground_truth[1]
    h_mask, w_mask, _ = np.shape(image)
    # mask
    mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)

    for seg in mask:
        bin_mask = maskUtils.decode(seg)
        bin_mask = np.logical_not(bin_mask)
        mask_miss = np.bitwise_and(mask_miss, bin_mask)

    ## image data augmentation
    # # randomly resize height and width independently, scale is changed
    # image, annos, mask_miss = tl.prepro.keypoint_random_resize(image, annos, mask_miss, zoom_range=(0.8, 1.2))# removed hao
    # # random rotate
    # image, annos, mask_miss = tl.prepro.keypoint_random_rotate(image, annos, mask_miss, rg=15.0)# removed hao
    # # random left-right flipping
    # image, annos, mask_miss = tl.prepro.keypoint_random_flip(image, annos, mask_miss, prob=0.5)# removed hao

    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
    # M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5) # hao removed: bug, keypoints will have error
    M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
    # M_shear = tl.prepro.affine_shear_matrix(x_shear=(-0.1, 0.1), y_shear=(-0.1, 0.1))
    M_combined = M_rotate.dot(M_zoom)
    # M_combined = M_rotate.dot(M_flip).dot(M_zoom)#.dot(M_shear)
    # M_combined = tl.prepro.affine_zoom_matrix(zoom_range=0.9) # for debug
    h, w, _ = image.shape
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)
    image = tl.prepro.affine_transform_cv2(image, transform_matrix)
    mask_miss = tl.prepro.affine_transform_cv2(mask_miss, transform_matrix, border_mode='replicate')
    annos = tl.prepro.affine_transform_keypoints(annos, transform_matrix)

    # random resize height and width together
    # image, annos, mask_miss = tl.prepro.keypoint_random_resize_shortestedge(
    #     image, annos, mask_miss, min_size=(hin, win), zoom_range=(0.95, 1.6)) # removed hao
    # random crop
    # image, annos, mask_miss = tl.prepro.keypoint_random_crop(image, annos, mask_miss, size=(hin, win))  # with padding # removed hao

    image, annos, mask_miss = tl.prepro.keypoint_random_flip(image, annos, mask_miss, prob=0.5)
    image, annos, mask_miss = tl.prepro.keypoint_resize_random_crop(image, annos, mask_miss, size=(hin, win)) # hao add

    # generate result maps including keypoints heatmap, pafs and mask
    h, w, _ = np.shape(image)
    height, width, _ = np.shape(image)
    heatmap = get_heatmap(annos, height, width)
    vectormap = get_vectormap(annos, height, width)
    resultmap = np.concatenate((heatmap, vectormap), axis=2)

    image = np.array(image, dtype=np.float32)

    img_mask = mask_miss.reshape(hin, win, 1)
    image = image * np.repeat(img_mask, 3, 2)

    resultmap = np.array(resultmap, dtype=np.float32)
    mask_miss = cv2.resize(mask_miss, (hout, wout), interpolation=cv2.INTER_AREA)
    mask_miss = np.array(mask_miss, dtype=np.float32)
    return image, resultmap, mask_miss


def _map_fn(img_list, annos):
    """TF Dataset pipeline."""
    image = tf.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Affine transform and get paf maps
    image, resultmap, mask = tf.py_func(_data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32])

    image = tf.reshape(image, [hin, win, 3])
    resultmap = tf.reshape(resultmap, [hout, wout, n_pos * 3])
    mask = tf.reshape(mask, [hout, wout, 1])

    image = tf.image.random_brightness(image, max_delta=45./255.)   # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)   # lower=0.2, upper=1.8)  caffe 0.3~1.5
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_hue(image, max_delta=0.1)
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return image, resultmap, mask


def single_train(training_dataset):
    ds = training_dataset.shuffle(buffer_size=4096)  # shuffle before loading images
    ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=multiprocessing.cpu_count() // 2)  # decouple the heavy map_fn
    ds = ds.batch(batch_size)  # TODO: consider using tf.contrib.map_and_batch
    ds = ds.prefetch(2)
    iterator = ds.make_one_shot_iterator()
    one_element = iterator.get_next()
    net, total_loss, log_tensors = make_model(*one_element, is_train=True, reuse=False)
    x_ = net.img  # net input
    last_conf = net.last_conf  # net output
    last_paf = net.last_paf  # net output
    confs_ = net.confs  # GT
    pafs_ = net.pafs  # GT
    mask = net.m1  # mask1, GT
    # net.m2 = m2                 # mask2, GT
    stage_losses = net.stage_losses
    l2_loss = net.l2_loss

    global_step = tf.Variable(1, trainable=False)
    print('Start - n_step: {} batch_size: {} lr_init: {} lr_decay_every_step: {}'.format(
        n_step, batch_size, lr_init, lr_decay_every_step))
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    opt = tf.train.MomentumOptimizer(lr_v, 0.9)
    train_op = opt.minimize(total_loss, global_step=global_step)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # start training
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # restore pre-trained weights
        try:
            # tl.files.load_and_assign_npz(sess, os.path.join(model_path, 'pose.npz'), net)
            tl.files.load_and_assign_npz_dict(sess=sess, name=os.path.join(model_path, 'pose.npz'))
        except:
            print("no pre-trained model")

        # train until the end
        sess.run(tf.assign(lr_v, lr_init))
        while True:
            tic = time.time()
            step = sess.run(global_step)
            if step != 0 and (step % lr_decay_every_step == 0):
                new_lr_decay = lr_decay_factor**(step // lr_decay_every_step)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))

            [_, _loss, _stage_losses, _l2, conf_result, paf_result] = \
                sess.run([train_op, total_loss, stage_losses, l2_loss, last_conf, last_paf])

            # tstring = time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
            lr = sess.run(lr_v)
            print('Total Loss at iteration {} / {} is: {} Learning rate {:10e} l2_loss {:10e} Took: {}s'.format(
                step, n_step, _loss, lr, _l2,
                time.time() - tic))
            for ix, ll in enumerate(_stage_losses):
                print('Network#', ix, 'For Branch', ix % 2 + 1, 'Loss:', ll)

            # save intermediate results and model
            if (step != 0) and (step % save_interval == 0):
                # save some results
                [img_out, confs_ground, pafs_ground, conf_result, paf_result,
                 mask_out] = sess.run([x_, confs_, pafs_, last_conf, last_paf, mask])
                draw_results(img_out, confs_ground, conf_result, pafs_ground, paf_result, mask_out, 'train_%d_' % step)
                # save model
                # tl.files.save_npz(
                #    net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'), sess=sess)
                # tl.files.save_npz(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)
                tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'), sess=sess)
                tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)
            if step == n_step:  # training finished
                break


def _mock_map_fn(img_list, annos):
    """TF Dataset pipeline."""
    image = tf.read_file(img_list)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    image = np.ones((hin, win, 3), dtype=np.float32)
    resultmap = np.ones((hout, wout, 57), dtype=np.float32)
    mask = np.ones((hout, wout, 1), dtype=np.float32)

    return image, resultmap, mask


def _parallel_train_model(img, results, mask):
    net, total_loss, log_tensors = make_model(img, results, mask, is_train=True, reuse=False)
    return net, total_loss, log_tensors


def parallel_train(training_dataset, kungfu_option):
    from kungfu import current_cluster_size, current_rank
    from kungfu.tensorflow.optimizers import SynchronousSGDOptimizer, SynchronousAveragingOptimizer, PairAveragingOptimizer

    ds = training_dataset.shuffle(buffer_size=4096)
    ds = ds.shard(num_shards=current_cluster_size(), index=current_rank())
    ds = ds.repeat(n_epoch)
    ds = ds.map(_map_fn, num_parallel_calls=4)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=1)

    iterator = ds.make_one_shot_iterator()
    one_element = iterator.get_next()
    net, total_loss, log_tensors = make_model(*one_element, is_train=True, reuse=False)
    x_ = net.img  # net input
    last_conf = net.last_conf  # net output
    last_paf = net.last_paf  # net output
    confs_ = net.confs  # GT
    pafs_ = net.pafs  # GT
    mask = net.m1  # mask1, GT
    # net.m2 = m2                 # mask2, GT
    stage_losses = net.stage_losses
    l2_loss = net.l2_loss

    global_step = tf.Variable(1, trainable=False)
    # scaled_lr = lr_init * current_cluster_size()  # Horovod: scale the learning rate linearly
    scaled_lr = lr_init  # Linear scaling rule is not working in openpose training.
    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(scaled_lr, trainable=False)

    opt = tf.train.MomentumOptimizer(lr_v, 0.9)

    # KungFu
    if kungfu_option == 'sync-sgd':
        opt = SynchronousSGDOptimizer(opt)
    elif kungfu_option == 'async-sgd':
        opt = PairAveragingOptimizer(opt)
    elif kungfu_option == 'sma':
        opt = SynchronousAveragingOptimizer(opt)
    else:
        raise RuntimeError('Unknown distributed training optimizer.')

    train_op = opt.minimize(total_loss, global_step=global_step)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # KungFu
    from kungfu.tensorflow.initializer import BroadcastGlobalVariablesOp
    bcast = BroadcastGlobalVariablesOp()

    global n_step, lr_decay_every_step
    n_step = n_step // current_cluster_size() + 1  # KungFu
    lr_decay_every_step = lr_decay_every_step // current_cluster_size() + 1  # KungFu

    # Start training
    with tf.Session(config=config) as sess:
        init.run()
        bcast.run()  # KungFu
        print('Worker{}: Initialized'.format(current_rank()))
        print('Worker{}: Start - n_step: {} batch_size: {} lr_init: {} lr_decay_every_step: {}'.format(
            current_rank(), n_step, batch_size, lr_init, lr_decay_every_step))

        # restore pre-trained weights
        try:
            # tl.files.load_and_assign_npz(sess, os.path.join(model_path, 'pose.npz'), net)
            tl.files.load_and_assign_npz_dict(sess=sess, name=os.path.join(model_path, 'pose.npz'))
        except:
            print("no pre-trained model")

        # train until the end
        while True:
            step = sess.run(global_step)
            if step == n_step:
                break

            tic = time.time()
            if step != 0 and (step % lr_decay_every_step == 0):
                new_lr_decay = lr_decay_factor**(step // lr_decay_every_step)
                sess.run(tf.assign(lr_v, scaled_lr * new_lr_decay))

            [_, _loss, _stage_losses, _l2, conf_result, paf_result] = \
                sess.run([train_op, total_loss, stage_losses, l2_loss, last_conf, last_paf])

            # tstring = time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
            lr = sess.run(lr_v)
            print(
                'Worker{}: Total Loss at iteration {} / {} is: {} Learning rate {:10e} l2_loss {:10e} Took: {}s'.format(
                    current_rank(), step, n_step, _loss, lr, _l2,
                    time.time() - tic))
            for ix, ll in enumerate(_stage_losses):
                print('Worker{}:', current_rank(), 'Network#', ix, 'For Branch', ix % 2 + 1, 'Loss:', ll)

            # save intermediate results and model
            if current_rank() == 0:  # KungFu
                if (step != 0) and (step % save_interval == 0):
                    # save some results
                    [img_out, confs_ground, pafs_ground, conf_result, paf_result,
                     mask_out] = sess.run([x_, confs_, pafs_, last_conf, last_paf, mask])
                    draw_results(img_out, confs_ground, conf_result, pafs_ground, paf_result, mask_out,
                                 'train_%d_' % step)

                    # save model
                    # tl.files.save_npz(
                    #    net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'), sess=sess)
                    # tl.files.save_npz(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)
                    tl.files.save_npz_dict(
                        net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'), sess=sess)
                    tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenPose-plus.')
    parser.add_argument('--kf-optimizer',
                        type=str,
                        default='sma',
                        help='available options: sync-sgd, async-sgd, sma')
    parser.add_argument('--parallel',
                        action='store_true',
                        default=False,
                        help='enable parallel training')
    args = parser.parse_args()

    if 'coco' in config.DATA.train_data:
        # automatically download MSCOCO data to "data/mscoco..."" folder
        train_im_path, train_ann_path, val_im_path, val_ann_path, _, _ = \
            load_mscoco_dataset(config.DATA.data_path, config.DATA.coco_version, task='person')

        # read coco training images contains valid people
        train_imgs_file_list, train_objs_info_list, train_mask_list, train_targets = \
            get_pose_data_list(train_im_path, train_ann_path)

        # read coco validating images contains valid people (you can use it for training as well)
        val_imgs_file_list, val_objs_info_list, val_mask_list, val_targets = \
            get_pose_data_list(val_im_path, val_ann_path)

    if 'custom' in config.DATA.train_data:
        ## read your own images contains valid people
        ## 1. if you only have one folder as follow:
        ##   data/your_data
        ##           /images
        ##               0001.jpeg
        ##               0002.jpeg
        ##           /coco.json
        # your_imgs_file_list, your_objs_info_list, your_mask_list, your_targets = \
        #     get_pose_data_list(config.DATA.your_images_path, config.DATA.your_annos_path)
        ## 2. if you have a folder with many folders: (which is common in industry)
        folder_list = tl.files.load_folder_list(path='data/your_data')
        your_imgs_file_list, your_objs_info_list, your_mask_list = [], [], []
        for folder in folder_list:
            _imgs_file_list, _objs_info_list, _mask_list, _targets = \
                get_pose_data_list(os.path.join(folder, 'images'), os.path.join(folder, 'coco.json'))
            print(len(_imgs_file_list))
            your_imgs_file_list.extend(_imgs_file_list)
            your_objs_info_list.extend(_objs_info_list)
            your_mask_list.extend(_mask_list)
        print("number of own images found:", len(your_imgs_file_list))

    # choose dataset for training
    if config.DATA.train_data == 'coco':
        # 1. only coco training set
        imgs_file_list = train_imgs_file_list
        train_targets = list(zip(train_objs_info_list, train_mask_list))
    elif config.DATA.train_data == 'custom':
        # 2. only your own data
        imgs_file_list = your_imgs_file_list
        train_targets = list(zip(your_objs_info_list, your_mask_list))
    elif config.DATA.train_data == 'coco_and_custom':
        # 3. your own data and coco training set
        imgs_file_list = train_imgs_file_list + your_imgs_file_list
        train_targets = list(zip(train_objs_info_list + your_objs_info_list, train_mask_list + your_mask_list))
    else:
        raise Exception('please choose a valid config.DATA.train_data setting.')

    # define data augmentation
    def generator():
        """TF Dataset generator."""
        assert len(imgs_file_list) == len(train_targets)
        for _input, _target in zip(imgs_file_list, train_targets):
            yield _input.encode('utf-8'), cPickle.dumps(_target)

    n_epoch = math.ceil(n_step / (len(imgs_file_list) / batch_size))
    dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))

    if args.parallel:
        parallel_train(dataset, args.kf_optimizer)
    else:
        single_train(dataset)
