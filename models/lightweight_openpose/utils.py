# -*- coding: utf-8 -*-

## xxx

import os
import cv2
import math
import logging
import numpy as np
import tensorflow as tf
from tensorlayer import logging
from tensorlayer.files.utils import (del_file, folder_exists, maybe_download_and_extract)

import matplotlib.pyplot as plt
from distutils.dir_util import mkpath
from scipy.spatial.distance import cdist
from pycocotools.coco import COCO, maskUtils

def get_heatmap(annos, height, width, hout, wout, n_pos):
    """

    Parameters
    -----------


    Returns
    --------

    """
    # n_pos is 19 for coco, 15 for MPII
    # the heatmap for every joints takes the maximum over all people
    joints_heatmap = np.zeros((n_pos, height, width), dtype=np.float32)

    # among all people
    for joint in annos:
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum

        for i, points in enumerate(joint):
            if points[0] < 0 or points[1] < 0:
                continue
            joints_heatmap = put_heatmap(joints_heatmap, i, points, 8.0)

    # 0: joint index, 1:y, 2:x
    joints_heatmap = joints_heatmap.transpose((1, 2, 0))

    # background
    joints_heatmap[:, :, -1] = np.clip(1 - np.amax(joints_heatmap, axis=2), 0.0, 1.0)

    #resize
    resized_joints_heatmap=np.zeros((hout,wout,n_pos),dtype=np.float32)
    for i in range(0, n_pos):
        resized_joints_heatmap[:,:,i] = cv2.resize(np.array(joints_heatmap[:, :, i]), (hout, wout))
    return resized_joints_heatmap

def put_heatmap(heatmap, plane_idx, center, sigma):
    """

    Parameters
    -----------


    Returns
    --------


    """
    center_x, center_y = center
    _, height, width = heatmap.shape[:3]

    th = 4.6052
    delta = math.sqrt(th * 2)

    x0 = int(max(0, center_x - delta * sigma + 0.5))
    y0 = int(max(0, center_y - delta * sigma + 0.5))

    x1 = int(min(width - 1, center_x + delta * sigma + 0.5))
    y1 = int(min(height - 1, center_y + delta * sigma + 0.5))

    exp_factor = 1 / 2.0 / sigma / sigma

    ## fast - vectorize
    arr_heatmap = heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1]
    y_vec = (np.arange(y0, y1 + 1) - center_y)**2  # y1 included
    x_vec = (np.arange(x0, x1 + 1) - center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > th] = 0
    heatmap[plane_idx, y0:y1 + 1, x0:x1 + 1] = np.maximum(arr_heatmap, arr_exp)
    return heatmap


def get_vectormap(annos, height, width , hout, wout, n_pos):
    """

    Parameters
    -----------


    Returns
    --------


    """
    n_pos = 19

    limb = list(
        zip([2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]))

    vectormap = np.zeros((n_pos * 2, height, width), dtype=np.float32)
    counter = np.zeros((n_pos, height, width), dtype=np.int16)

    for joint in annos:
        if len(joint) != 19:
            print('THE LENGTH IS NOT 19 ERROR:', len(joint))
        for i, (a, b) in enumerate(limb):
            a -= 1
            b -= 1

            v_start = joint[a]
            v_end = joint[b]
            # exclude invisible or unmarked point
            if v_start[0] < -100 or v_start[1] < -100 or v_end[0] < -100 or v_end[1] < -100:
                continue
            vectormap = cal_vectormap_fast(vectormap, counter, i, v_start, v_end)

    # normalize the PAF (otherwise longer limb gives stronger absolute strength)
    for i in range(0,n_pos):
        filter_counter=np.where(counter[i]<=0,1,0)
        div_counter=filter_counter+(1-filter_counter)*counter[i]
        vectormap[i*2+0]/=div_counter
        vectormap[i*2+1]/=div_counter

    #resize
    resized_vectormap=np.zeros((n_pos*2,hout,wout),dtype= np.float32)
    for i in range(0, n_pos * 2):
        resized_vectormap[i,:,:] = cv2.resize(np.array(vectormap[i, :, :]), (hout, wout), interpolation=cv2.INTER_AREA)
    resized_vectormap=resized_vectormap.transpose(1,2,0)
    return resized_vectormap

def cal_vectormap_ori(vectormap, countmap, i, v_start, v_end):
    """

    Parameters
    -----------


    Returns
    --------


    """
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]
    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            bec_x = x - v_start[0]
            bec_y = y - v_start[1]
            dist = abs(bec_x * norm_y - bec_y * norm_x)

            # orthogonal distance is < then threshold
            if dist > threshold:
                continue
            countmap[i][y][x] += 1
            vectormap[i * 2 + 0][y][x] = norm_x
            vectormap[i * 2 + 1][y][x] = norm_y

    return vectormap


def cal_vectormap_fast(vectormap, countmap, i, v_start, v_end):
    """

    Parameters
    -----------


    Returns
    --------


    """
    _, height, width = vectormap.shape[:3]
    _, height, width = vectormap.shape[:3]

    threshold = 8
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]

    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(min(v_start[0], v_end[0]) - threshold))
    min_y = max(0, int(min(v_start[1], v_end[1]) - threshold))

    max_x = min(width, int(max(v_start[0], v_end[0]) + threshold))
    max_y = min(height, int(max(v_start[1], v_end[1]) + threshold))

    norm_x = vector_x / length
    norm_y = vector_y / length

    x_vec = (np.arange(min_x, max_x) - v_start[0]) * norm_y
    y_vec = (np.arange(min_y, max_y) - v_start[1]) * norm_x

    xv, yv = np.meshgrid(x_vec, y_vec)

    dist_matrix = abs(xv - yv)
    filter_matrix = np.where(dist_matrix > threshold, 0, 1)
    countmap[i, min_y:max_y, min_x:max_x] += filter_matrix

    ori_vecx_map=vectormap[i*2+0,min_y:max_y,min_x:max_x]
    vectormap[i*2+0,min_y:max_y,min_x:max_x]=ori_vecx_map*(1-filter_matrix)+norm_x*filter_matrix
    ori_vecy_map=vectormap[i*2+1,min_y:max_y,min_x:max_x]
    vectormap[i*2+1,min_y:max_y,min_x:max_x]=ori_vecy_map*(1-filter_matrix)+norm_y*filter_matrix
    return vectormap


def draw_results(images, heats_ground, heats_result, pafs_ground, pafs_result, masks, save_dir ,name=''):
    """Save results for debugging.

    Parameters
    -----------
    images : a list of RGB images
    heats_ground : a list of keypoint heat maps or None
    heats_result : a list of keypoint heat maps or None
    pafs_ground : a list of paf vector maps or None
    pafs_result : a list of paf vector maps or None
    masks : a list of mask for people
    """
    # interval = len(images)
    _,_,_,n_pos=heats_ground.shape
    for i in range(len(images)):
        if heats_ground is not None:
            heat_ground = heats_ground[i]
        if heats_result is not None:
            heat_result = heats_result[i]
        if pafs_ground is not None:
            paf_ground = pafs_ground[i]
        if pafs_result is not None:
            paf_result = pafs_result[i]
        if masks is not None:
            # print(masks.shape)
            mask = masks[i, :, :, 0]
            # print(mask.shape)
            mask = mask[:, :, np.newaxis]
            # mask = masks[:,:,:,0]
            # mask = mask.reshape(hout, wout, 1)
            mask1 = np.repeat(mask, n_pos, 2)
            mask2 = np.repeat(mask, n_pos * 2, 2)
            # print(mask1.shape, mask2.shape)
        image = images[i]

        fig = plt.figure(figsize=(8, 8))
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(image)

        if pafs_ground is not None:
            a = fig.add_subplot(2, 3, 2)
            a.set_title('Vectormap_ground')
            vectormap = paf_ground * mask2
            tmp2 = vectormap.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

            # tmp2_odd = tmp2_odd * 255
            # tmp2_odd = tmp2_odd.astype(np.int)
            plt.imshow(tmp2_odd, alpha=0.3)

            # tmp2_even = tmp2_even * 255
            # tmp2_even = tmp2_even.astype(np.int)
            plt.colorbar()
            plt.imshow(tmp2_even, alpha=0.3)

        if pafs_result is not None:
            a = fig.add_subplot(2, 3, 3)
            a.set_title('Vectormap result')
            if masks is not None:
                vectormap = paf_result * mask2
            else:
                vectormap = paf_result
            tmp2 = vectormap.transpose((2, 0, 1))
            tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
            tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)
            plt.imshow(tmp2_odd, alpha=0.3)

            plt.colorbar()
            plt.imshow(tmp2_even, alpha=0.3)

        if heats_result is not None:
            a = fig.add_subplot(2, 3, 4)
            a.set_title('Heatmap result')
            if masks is not None:
                heatmap = heat_result * mask1
            else:
                heatmap = heat_result
            tmp = heatmap
            tmp = np.amax(heatmap[:, :, :-1], axis=2)

            plt.colorbar()
            plt.imshow(tmp, alpha=0.3)

        if heats_ground is not None:
            a = fig.add_subplot(2, 3, 5)
            a.set_title('Heatmap ground truth')
            if masks is not None:
                heatmap = heat_ground * mask1
            else:
                heatmap = heat_ground
            tmp = heatmap
            tmp = np.amax(heatmap[:, :, :-1], axis=2)

            plt.colorbar()
            plt.imshow(tmp, alpha=0.3)

        if masks is not None:
            a = fig.add_subplot(2, 3, 6)
            a.set_title('Mask')
            # print(mask.shape, tmp.shape)
            plt.colorbar()
            plt.imshow(mask[:, :, 0], alpha=0.3)
        # plt.savefig(str(i)+'.png',dpi=300)
        # plt.show()
        plt.savefig(os.path.join(save_dir, '%s%d.png' % (name, i)), dpi=300)



def vis_annos(image, annos, save_dir ,name=''):
    """Save results for debugging.

    Parameters
    -----------
    images : single RGB image
    annos  : annotation, list of lists
    """

    fig = plt.figure(figsize=(8, 8))
    a = fig.add_subplot(1, 1, 1)

    plt.imshow(image)
    for people in annos:
        for idx, jo in enumerate(people):
            if jo[0] > 0 and jo[1] > 0:
                plt.plot(jo[0], jo[1], '*')

    plt.savefig(os.path.join(save_dir, 'keypoints%s%d.png' % (name, i)), dpi=300)


def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor
