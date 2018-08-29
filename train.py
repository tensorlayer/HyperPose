#!/usr/bin/env python3

import os
import time
import math
import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np

import _pickle as cPickle
import tensorflow as tf
import tensorlayer as tl
from config import config
from models import model
from pycocotools.coco import maskUtils
from tensorlayer.prepro import (keypoint_random_crop, keypoint_random_flip, keypoint_random_resize,
                                keypoint_random_resize_shortestedge, keypoint_random_rotate)
from utils import (PoseInfo, draw_results, get_heatmap, get_vectormap, load_mscoco_dataset)

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

tl.files.exists_or_mkdir(config.LOG.vis_path, verbose=False)  # to save visualization results
tl.files.exists_or_mkdir(config.MODEL.model_path, verbose=False)  # to save model files

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# define hyper-parameters for training
batch_size = config.TRAIN.batch_size
# n_epoch = config.TRAIN.n_epoch
decay_every_step = config.TRAIN.decay_every_step
n_step = config.TRAIN.n_step
save_interval = config.TRAIN.save_interval
weight_decay = config.TRAIN.weight_decay
base_lr = config.TRAIN.base_lr
gamma = config.TRAIN.gamma

# define hyper-parameters for model
model_path = config.MODEL.model_path
n_pos = config.MODEL.n_pos
hin = config.MODEL.hin
win = config.MODEL.win
hout = config.MODEL.hout
wout = config.MODEL.wout


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
    # randomly resize height and width independently, scale is changed
    image, annos, mask_miss = keypoint_random_resize(image, annos, mask_miss, zoom_range=(0.8, 1.2))
    # random rotate
    image, annos, mask_miss = keypoint_random_rotate(image, annos, mask_miss, rg=15.0)
    # random left-right flipping
    image, annos, mask_miss = keypoint_random_flip(image, annos, mask_miss, prob=0.5)
    # random resize height and width together
    image, annos, mask_miss = keypoint_random_resize_shortestedge(image, annos, mask_miss, min_size=(hin, win), zoom_range=(0.95, 1.6))
    # random crop
    image, annos, mask_miss = keypoint_random_crop(image, annos, mask_miss, size=(hin, win)) # with padding

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
    image, resultmap, mask = tf.py_func(_data_aug_fn, [image, annos], [tf.float32, tf.float32, tf.float32])
    return image, resultmap, mask


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


if __name__ == '__main__':

    ## automatically download MSCOCO data to "data/mscoco..."" folder
    train_im_path, train_ann_path, val_im_path, val_ann_path, _, _ = \
        load_mscoco_dataset(config.DATA.data_path, config.DATA.coco_version, task='person')

    ## read coco training images contains valid people
    train_imgs_file_list, train_objs_info_list, train_mask_list, train_targets = \
        get_pose_data_list(train_im_path, train_ann_path)

    ## read coco validating images contains valid people (you can use it for training as well)
    val_imgs_file_list, val_objs_info_list, val_mask_list, val_targets = \
        get_pose_data_list(val_im_path, val_ann_path)

    ## read your own images contains valid people
    ## 1. if you only have one folder as follow:
    #   data/your_data
    #           /images
    #               0001.jpeg
    #               0002.jpeg
    #           /coco.json
    # your_imgs_file_list, your_objs_info_list, your_mask_list, your_targets = \
    #     get_pose_data_list(config.DATA.your_images_path, config.DATA.your_annos_path)
    ## 2. if you have a folder with many folders: (which is common in industry)
    # folder_list = tl.files.load_folder_list(path='data/your_data')
    # your_imgs_file_list, your_objs_info_list, your_mask_list = [], [], []
    # for folder in folder_list:
    #     _imgs_file_list, _objs_info_list, _mask_list, _targets = \
    #         get_pose_data_list(os.path.join(folder, 'images'), os.path.join(folder, 'coco.json'))
    #     print(len(_imgs_file_list))
    #     your_imgs_file_list.extend(_imgs_file_list)
    #     your_objs_info_list.extend(_objs_info_list)
    #     your_mask_list.extend(_mask_list)
    # print("number of own images found:", len(your_imgs_file_list))

    ## choice dataset for training
    ## 1. only coco training set
    imgs_file_list = train_imgs_file_list
    train_targets = list(zip(train_objs_info_list, train_mask_list))
    ## 2. your own data and coco training set
    # imgs_file_list = train_imgs_file_list + your_imgs_file_list
    # train_targets = list(zip(train_objs_info_list + your_objs_info_list, train_mask_list + your_mask_list))
    ## 3. only your own data
    # imgs_file_list = your_imgs_file_list
    # train_targets = list(zip(your_objs_info_list, your_mask_list))

    # define data augmentation
    def generator():
        """TF Dataset generartor."""
        assert len(imgs_file_list) == len(train_targets)
        for _input, _target in zip(imgs_file_list, train_targets):
            yield _input.encode('utf-8'), cPickle.dumps(_target)

    n_epoch = math.ceil(n_step / len(imgs_file_list))
    dataset = tf.data.Dataset().from_generator(generator, output_types=(tf.string, tf.string))
    dataset = dataset.map(_map_fn, num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=2046)
    dataset = dataset.repeat(n_epoch)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=20)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    if config.TRAIN.train_mode == 'placeholder':
        ## Train with placeholder can help your to check the data easily.
        ## define model architecture
        x = tf.placeholder(tf.float32, [None, hin, win, 3], "image")
        confs = tf.placeholder(tf.float32, [None, hout, wout, n_pos], "confidence_maps")
        pafs = tf.placeholder(tf.float32, [None, hout, wout, n_pos * 2], "pafs")
        # if the people does not have keypoints annotations, ignore the area
        img_mask1 = tf.placeholder(tf.float32, [None, hout, wout, n_pos], 'img_mask1')
        img_mask2 = tf.placeholder(tf.float32, [None, hout, wout, n_pos * 2], 'img_mask2')
        # num_images = np.shape(imgs_file_list)[0]

        cnn, b1_list, b2_list, net = model(x, n_pos, img_mask1, img_mask2, True, False)

        ## define loss
        losses = []
        last_losses_l1 = []
        last_losses_l2 = []
        stage_losses = []
        L2 = 0.0
        for idx, (l1, l2) in enumerate(zip(b1_list, b2_list)):
            loss_l1 = tf.nn.l2_loss((tf.concat(l1.outputs, axis=0) - tf.concat(confs, axis=0)) * img_mask1)
            loss_l2 = tf.nn.l2_loss((tf.concat(l2.outputs, axis=0) - tf.concat(pafs, axis=0)) * img_mask2)
            losses.append(tf.reduce_mean([loss_l1, loss_l2]))
            stage_losses.append(loss_l1 / batch_size)
            stage_losses.append(loss_l2 / batch_size)
        last_losses_l1.append(loss_l1)
        last_losses_l2.append(loss_l2)
        last_conf = b1_list[-1].outputs
        last_paf = b2_list[-1].outputs

        for p in tl.layers.get_variables_with_name('kernel', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.0005)(p)
        total_loss = tf.reduce_sum(losses) / batch_size + L2

        global_step = tf.Variable(1, trainable=False)
        print('Start - n_step: {} batch_size: {} base_lr: {} decay_every_step: {}'.format(
            n_step, batch_size, base_lr, decay_every_step))
        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(base_lr, trainable=False)

        opt = tf.train.MomentumOptimizer(lr_v, 0.9)
        train_op = opt.minimize(total_loss, global_step=global_step)
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

        ## start training
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            ## restore pretrained weights  TODO: use tl.models.VGG19
            # npy_file = np.load('models', encoding='latin1').item()
            # params = []
            # for val in sorted(npy_file.items()):
            #     if val[0] == 'conv4_3':
            #         break
            #     W = np.asarray(val[1][0])
            #     b = np.asarray(val[1][1])
            #     print("Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            #     params.extend([W, b])
            # tl.files.assign_params(sess, params, cnn)
            # print("Restoring model from npy file")
            # cnn.restore_params(sess)

            ## train until the end
            sess.run(tf.assign(lr_v, base_lr))
            while (True):
                tic = time.time()
                step = sess.run(global_step)
                if step != 0 and (step % decay_every_step == 0):
                    new_lr_decay = gamma**(step // decay_every_step)
                    sess.run(tf.assign(lr_v, base_lr * new_lr_decay))

                # get a batch of training data. TODO change to direct feed without using placeholder
                tran_batch = sess.run(one_element)
                # get image
                x_ = tran_batch[0]
                # get conf and paf maps
                map_batch = tran_batch[1]
                confs_ = map_batch[:, :, :, 0:n_pos] # 0:19
                pafs_ = map_batch[:, :, :, n_pos::]  # 19:57
                # get mask
                mask = tran_batch[2]
                mask = mask.reshape(batch_size, hout, wout, 1)
                mask1 = np.repeat(mask, n_pos, 3)
                mask2 = np.repeat(mask, n_pos * 2, 3)

                ## save some training data for debugging data augmentation (slow)
                # draw_results(x_, confs_, None, pafs_, None, mask, 'check_batch_{}_'.format(step))

                [_, the_loss, loss_ll, L2_reg, conf_result, weight_norm, paf_result] = sess.run(
                    [train_op, total_loss, stage_losses, L2, last_conf, L2, last_paf],
                    feed_dict={
                        x: x_,
                        confs: confs_,
                        pafs: pafs_,
                        img_mask1: mask1,
                        img_mask2: mask2
                    })

                # tstring = time.strftime('%d-%m %H:%M:%S', time.localtime(time.time()))
                lr = sess.run(lr_v)
                print('Total Loss at iteration {} / {} is: {} Learning rate {:10e} weight_norm {:10e} Took: {}s'.format(
                    step, n_step, the_loss, lr, weight_norm, tic-time.time()))
                for ix, ll in enumerate(loss_ll):
                    print('Network#', ix, 'For Branch', ix % 2 + 1, 'Loss:', ll)

                ## save intermedian results and model
                if (step != 0) and (step % save_interval == 0):
                    draw_results(x_, confs_, conf_result, pafs_, paf_result, mask, 'train_%d_' % step)
                    tl.files.save_npz_dict(
                        net.all_params, os.path.join(model_path, 'pose' + str(step) + '.npz'), sess=sess)
                    tl.files.save_npz_dict(net.all_params, os.path.join(model_path, 'pose.npz'), sess=sess)
                if step == n_step:  # training finished
                    break

    elif config.TRAIN.train_mode == 'datasetapi':  # TODO
        ## Train with TensorFlow dataset mode is usually faster than placeholder.
        raise Exception("TODO")
    elif config.TRAIN.train_mode == 'distributed':  # TODO
        ## Train with distributed mode.
        raise Exception("TODO tl.distributed.Trainer")
