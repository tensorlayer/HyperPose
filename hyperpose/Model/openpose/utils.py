# -*- coding: utf-8 -*-

## xxx

import os
import cv2
import math
import logging
import numpy as np
import tensorflow as tf
from tensorlayer.files.utils import (del_file, folder_exists, maybe_download_and_extract)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distutils.dir_util import mkpath
from scipy.spatial.distance import cdist
from ..human import Human
from ..common import tf_repeat,TRAIN,MODEL,DATA


def preprocess(annos,img_height,img_width,model_hout,model_wout,parts,limbs,data_format="channels_first"):
    '''preprocess function of openpose class models
    
    take keypoints annotations, image height and width, model input height and width, and dataset type,
    return the constructed conf_map and paf_map

    Parameters
    ----------
    arg1 : list
        a list of annotation, each annotation is a list of keypoints that belongs to a person, each keypoint follows the
        format (x,y), and x<0 or y<0 if the keypoint is not visible or not annotated.
        the annotations must from a known dataset_type, other wise the keypoint and limbs order will not be correct.
    
    arg2 : Int
        height of the input image, need this to make use of keypoint annotation
    
    arg3 : Int
        width of the input image, need this to make use of keypoint annotation

    arg4 : Int
        height of the model output, will be the height of the generated maps
    
    arg5 : Int
        width of the model output, will be the width of the generated maps

    arg6 : Config.DATA
        a enum value of enum class Config.DATA
        dataset_type where the input annotation list from, because to generate correct
        conf_map and paf_map, the order of keypoints and limbs should be awared.

    arg7 : string
        data format speficied for channel order
        available input:
        'channels_first': data_shape C*H*W
        'channels_last': data_shape H*W*C  

    Returns
    -------
    list
        including two element
        conf_map: heatmaps of keypoints, shape C*H*W(channels_first) or H*W*C(channels_last)
        paf_map: heatmaps of limbs, shape C*H*W(channels_first) or H*W*C(channels_last)
    '''
    heatmap=get_heatmap(annos,img_height,img_width,model_hout,model_wout,parts,limbs,data_format)
    vectormap=get_vectormap(annos,img_height,img_width,model_hout,model_wout,parts,limbs,data_format)
    return heatmap,vectormap

def postprocess(conf_map,paf_map,img_h,img_w,parts,limbs,data_format="channels_first",colors=None):
    '''postprocess function of openpose class models
    
    take model predicted feature maps, output parsed human objects, each one contains all detected keypoints of the person

    Parameters
    ----------
    arg1 : numpy array
        model predicted conf_map, heatmaps of keypoints, shape C*H*W(channels_first) or H*W*C(channels_last)
    
    arg2 : numpy array
        model predicted paf_map, heatmaps of limbs, shape C*H*W(channels_first) or H*W*C(channels_last)
    
    arg3 : Config.DATA
        an enum value of enum class Config.DATA
        width of the model output, will be the width of the generated maps

    arg4 : string
        data format speficied for channel order
        available input:
        'channels_first': data_shape C*H*W
        'channels_last': data_shape H*W*C 

    Returns
    -------
    list
        contain object of humans,see Model.Human for detail information of Human object
    '''
    from .processor import PostProcessor
    if(type(conf_map)!=np.ndarray):
        conf_map=conf_map.numpy()
    if(type(paf_map)!=np.ndarray):
        paf_map=paf_map.numpy()

    if(colors==None):
        colors=[[255,0,0]]*len(parts)
    post_processor=PostProcessor(parts,limbs,colors)
    humans=post_processor.process(conf_map,paf_map,img_h,img_w,data_format=data_format)
    return humans

def visualize(img,conf_map,paf_map,save_name="maps",save_dir="./save_dir/vis_dir",data_format="channels_first",save_tofile=True):
    '''visualize function of openpose class models
    
    take model predict feature maps, output visualized image.
    the image will be saved at 'save_dir'/'save_name'_visualize.png

    Parameters
    ----------
    arg1 : numpy array
        image

    arg2 : numpy array
        model output conf_map, heatmaps of keypoints, shape C*H*W(channels_first) or H*W*C(channels_last)
    
    arg3 : numpy array
        model output paf_map, heatmaps of limbs, shape C*H*W(channels_first) or H*W*C(channels_last)
    
    arg4 : String
        specify output image name to distinguish.

    arg5 : String
        specify which directory to save the visualized image.

    arg6 : string
        data format speficied for channel order
        available input:
        'channels_first': data_shape C*H*W
        'channels_last': data_shape H*W*C  

    Returns
    -------
    None
    '''
    if(type(img)!=np.ndarray):
        img=img.numpy()
    if(type(conf_map)!=np.ndarray):
        conf_map=conf_map.numpy()
    if(type(paf_map)!=np.ndarray):
        paf_map=paf_map.numpy()

    if(data_format=="channels_last"):
        conf_map=np.transpose(conf_map,[2,0,1])
        paf_map=np.tranpose(paf_map,[2,0,1])
    elif(data_format=="channels_first"):
        img=np.transpose(img,[1,2,0])
    os.makedirs(save_dir,exist_ok=True)
    ori_img=np.clip(img*255.0,0.0,255.0).astype(np.uint8)
    vis_img=ori_img.copy()
    fig=plt.figure(figsize=(8,8))
    #show input image
    a=fig.add_subplot(2,2,1)
    a.set_title("input image")
    plt.imshow(vis_img)
    #show conf_map
    show_conf_map=np.amax(np.abs(conf_map[:-1,:,:]),axis=0)
    a=fig.add_subplot(2,2,3)
    a.set_title("conf_map")
    plt.imshow(show_conf_map)
    #show paf_map
    show_paf_map=np.amax(np.abs(paf_map[:,:,:]),axis=0)
    a=fig.add_subplot(2,2,4)
    a.set_title("paf_map")
    plt.imshow(show_paf_map)
    #save
    if(save_tofile):
        plt.savefig(f"{save_dir}/{save_name}_visualize.png")
        plt.close('all')
    return show_conf_map,show_paf_map

def get_heatmap(annos, height, width, hout, wout, parts, limbs, data_format="channels_first"):
    """

    Parameters
    -----------


    Returns
    --------

    """
    # n_pos is 19 for coco, 15 for MPII
    # the heatmap for every joints takes the maximum over all people
    n_pos=len(parts)
    joints_heatmap = np.zeros((n_pos, hout, wout), dtype=np.float32)
    stride=height/hout
    # among all people
    for joint in annos:
        # generate heatmap for every keypoints
        # loop through all people and keep the maximum

        for i, point in enumerate(joint):
            if point[0] < 0 or point[1] < 0:
                continue
            joints_heatmap = put_heatmap(joints_heatmap, i, point, stride, 7.0)

    # 0: joint index, 1:y, 2:x
    joints_heatmap[-1, :, :] = np.clip(1 - np.amax(joints_heatmap, axis=0), 0.0, 1.0)

    #resize
    if(data_format=="channels_last"):
        joints_heatmap=np.transpose(joints_heatmap,[1,2,0])
    return joints_heatmap

def put_heatmap(heatmap, plane_idx, center, stride, sigma):
    """

    Parameters
    -----------


    Returns
    --------


    """
    center_x,center_y=center
    _,hout,wout=heatmap.shape[:3]

    thresh = 4.6052
    offset = stride/2-0.5
    exp_factor = 1/(2*sigma*sigma)

    y=np.arange(0,hout)*stride+offset
    x=np.arange(0,wout)*stride+offset
       
    # fast - vectorize
    # meshgrid(x,y)=>(shape_y*shape_x)
    y_vec=(y-center_y)**2
    x_vec=(x-center_x)**2
    xv, yv = np.meshgrid(x_vec, y_vec)
    arr_sum = exp_factor * (xv + yv)
    arr_exp = np.exp(-arr_sum)
    arr_exp[arr_sum > thresh] = 0
    heatmap[plane_idx,:,:]=np.maximum(heatmap[plane_idx,:,:],arr_exp)
    return heatmap


def get_vectormap(annos, height, width , hout, wout, parts, limbs, data_format="channels_first"):
    """

    Parameters
    -----------


    Returns
    --------


    """
    n_limbs=len(limbs)
    stride=height/hout
    vectormap = np.zeros((2*n_limbs, hout, wout), dtype=np.float32)
    counter = np.zeros((n_limbs, hout, wout), dtype=np.int16)


    for joint in annos:
        for i, (a, b) in enumerate(limbs):
            # exclude invisible or unmarked point
            if joint[a][0] < -100 or joint[a][1] < -100 or joint[b][0] < -100 or joint[b][1] < -100:
                continue
            v_start=np.array(joint[a])/stride
            v_end=np.array(joint[b])/stride
            vectormap = cal_vectormap_fast(vectormap, counter, i, v_start, v_end)

    # normalize the PAF (otherwise longer limb gives stronger absolute strength)
    for i in range(0,n_limbs):
        filter_counter=np.where(counter[i]<=0,1,0)
        div_counter=filter_counter+(1-filter_counter)*counter[i]
        vectormap[i*2+0]/=div_counter
        vectormap[i*2+1]/=div_counter

    #resize
    if(data_format=="channels_last"):
        vectormap=np.transpose(vectormap,[1,2,0])
    return vectormap

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
    _, hout, wout = vectormap.shape[:3]

    threshold = 1
    vector_x = v_end[0] - v_start[0]
    vector_y = v_end[1] - v_start[1]

    length = math.sqrt(vector_x**2 + vector_y**2)
    if length == 0:
        return vectormap

    min_x = max(0, int(np.round(min(v_start[0], v_end[0]) - threshold)))
    min_y = max(0, int(np.round(min(v_start[1], v_end[1]) - threshold)))

    max_x = min(wout, int(np.round(max(v_start[0], v_end[0]) + threshold)))
    max_y = min(hout, int(np.round(max(v_start[1], v_end[1]) + threshold)))

    norm_x = vector_x / length
    norm_y = vector_y / length

    x_vec = (np.arange(min_x, max_x) - v_start[0]) * norm_y
    y_vec = (np.arange(min_y, max_y) - v_start[1]) * norm_x

    xv, yv = np.meshgrid(x_vec, y_vec)

    dist_matrix = abs(xv - yv)
    filter_matrix = np.where(dist_matrix > threshold, 0, 1)
    countmap[i, min_y:max_y, min_x:max_x] += filter_matrix

    vectormap[i*2+0,min_y:max_y,min_x:max_x]+=norm_x*filter_matrix
    vectormap[i*2+1,min_y:max_y,min_x:max_x]+=norm_y*filter_matrix
    return vectormap


def draw_results(images, heats_ground, heats_result, pafs_ground, pafs_result, masks, save_dir ,name='', data_format="channels_first"):
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
    if(data_format=="channels_last"):
        images=np.transpose(images,[0,3,1,2])
        heats_ground=np.transpose(heats_ground,[0,3,1,2])
        heats_result=np.transpose(heats_result,[0,3,1,2])
        pafs_ground=np.transpose(pafs_ground,[0,3,1,2])
        pafs_result=np.transpose(pafs_result,[0,3,1,2])
        masks=np.transpose(masks,[0,3,1,2])

    _,n_confmaps,_,_=heats_ground.shape
    _,n_pafmaps,_,_=pafs_ground.shape
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
            mask = masks[i, 0, :, :]
            # print(mask.shape)
            mask = mask[np.newaxis, :, :]
            # mask = masks[:,:,:,0]
            # mask = mask.reshape(hout, wout, 1)
            mask1 = np.repeat(mask, n_confmaps, 0)
            mask2 = np.repeat(mask, n_pafmaps, 0)
            # print(mask1.shape, mask2.shape)
        image = images[i]

        image=image.transpose([1,2,0])
        img_h,img_w,img_c=image.shape
        fig = plt.figure(figsize=(8, 8))
        a = fig.add_subplot(2, 3, 1)
        plt.imshow(image)
        
        if pafs_ground is not None:
            a = fig.add_subplot(2, 3, 2)
            a.set_title('Vectormap_ground')
            vectormap=paf_ground
            if(masks is not None):
                vectormap = paf_ground * mask2
            tmp2=np.amax(np.absolute(vectormap[:, :, :]),axis=0)
            plt.imshow(tmp2, alpha=0.8)
            plt.colorbar()

        if pafs_result is not None:
            a = fig.add_subplot(2, 3, 3)
            a.set_title('Vectormap result')
            if masks is not None:
                vectormap = paf_result * mask2
            else:
                vectormap = paf_result
            tmp2 = vectormap
            tmp2=np.amax(np.absolute(vectormap[:, :, :]),axis=0)
            plt.imshow(tmp2, alpha=0.8)
            plt.colorbar()

        if heats_result is not None:
            a = fig.add_subplot(2, 3, 4)
            a.set_title('Heatmap result')
            if masks is not None:
                heatmap = heat_result * mask1
            else:
                heatmap = heat_result
            tmp = heatmap
            tmp = np.amax(heatmap[:-1, :, :], axis=0)
            
            plt.imshow(tmp, alpha=0.8)
            plt.colorbar()

        if heats_ground is not None:
            a = fig.add_subplot(2, 3, 5)
            a.set_title('Heatmap ground truth')
            if masks is not None:
                heatmap = heat_ground * mask1
            else:
                heatmap = heat_ground
            tmp = heatmap
            tmp = np.amax(heatmap[:-1, :, :], axis=0)
            
            plt.imshow(tmp, alpha=0.8)
            plt.colorbar()

        if masks is not None:
            a = fig.add_subplot(2, 3, 6)
            a.set_title('Mask')
            plt.imshow(mask[0, :, :], alpha=0.8)
            plt.colorbar()
            
        os.makedirs(save_dir,exist_ok=True)
        plt.savefig(os.path.join(save_dir, '%s_%d.png' % (name, i)), dpi=300)
        plt.close()



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



from .define import CocoPart,CocoLimb,CocoColor,Coco_flip_list
from .define import MpiiPart,MpiiLimb,MpiiColor,Mpii_flip_list

def get_parts(dataset_type):
    if(dataset_type==DATA.MSCOCO):
        return CocoPart
    elif(dataset_type==DATA.MPII):
        return MpiiPart
    else:
        return dataset_type.get_parts()

def get_limbs(dataset_type):
    if(dataset_type==DATA.MSCOCO):
        return CocoLimb
    elif(dataset_type==DATA.MPII):
        return MpiiLimb
    else:
        return dataset_type.get_limbs()

def get_colors(dataset_type):
    if(dataset_type==DATA.MSCOCO):
        return CocoColor
    elif(dataset_type==DATA.MPII):
        return MpiiColor
    else:
        return dataset_type.get_colors()

def get_flip_list(dataset_type):
    if(dataset_type==DATA.MSCOCO):
        return Coco_flip_list
    elif(dataset_type==DATA.MPII):
        return Mpii_flip_list