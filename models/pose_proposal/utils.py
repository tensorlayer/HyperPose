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

def get_pose_proposals(kpts_list,bbxs,hin,win,hout,wout,hnei,wnei,img_mask=None):
    limbs = list(
        zip([2, 9,  10,  2, 12, 13, 2, 3, 4,  2, 6, 7,  2, 1,  1,  15, 16, 19, 19, 19],
            [9, 10, 11, 12, 13, 14, 3, 4, 5,  6, 7, 8,  1, 15, 16, 17, 18, 1 , 17, 18]))
    K,L=18,20
    grid_x=win/wout
    grid_y=hin/hout
    delta=np.zeros(shape=(K+1,hout,wout))
    tx=np.zeros(shape=(K+1,hout,wout))
    ty=np.zeros(shape=(K+1,hout,wout))
    tw=np.zeros(shape=(K+1,hout,wout))
    th=np.zeros(shape=(K+1,hout,wout))
    te=np.zeros(shape=(L,hnei,wnei,hout,wout))
    te_mask=np.zeros(shape=(L,hnei,wnei,hout,wout))
    #generate pose proposals for each labels person
    for kpts,bbx in zip(kpts_list,bbxs):
        #change the background keypoint to the added human instance keypoint
        ins_x,ins_y,ins_w,ins_h=bbx
        center_x=ins_x+ins_w//2
        center_y=ins_y+ins_h//2
        kpts[-1]=[center_x,center_y]
        part_size=int(min(ins_w,ins_h)/5)
        for k,kpt in enumerate(kpts):
            x,y=kpt[0],kpt[1]
            if(x<0 or y<0):
                continue
            if(x>=win or y>=hin):
                continue
            if(img_mask.all()!=None):
                #joints are masked
                if(img_mask[int(x),int(y)]==0):
                    continue
            cx,cy=x/grid_x,y/grid_y
            ix,iy=int(cx),int(cy)
            delta[k,iy,ix]=1
            tx[k,iy,ix]=cx-ix
            ty[k,iy,ix]=cy-iy
            if(k!=len(kpts)-1):
                tw[k,iy,ix]=part_size/win
                th[k,iy,ix]=part_size/hin
            else:
                tw[k,iy,ix]=ins_w/win
                th[k,iy,ix]=ins_h/hin
        for l,(s_id,d_id) in enumerate(limbs):
            s_id-=1
            d_id-=1
            s_kpt,d_kpt=kpts[s_id],kpts[d_id]
            s_ix,s_iy=int(s_kpt[0]/grid_x),int(s_kpt[1]/grid_y)
            d_ix,d_iy=int(d_kpt[0]/grid_x),int(d_kpt[1]/grid_y)
            #generate te
            delta_ix=d_ix-s_ix+wnei//2
            delta_iy=d_iy-s_iy+hnei//2
            if(s_ix<0 or s_ix>=wout or s_iy<0 or s_iy>=hout):
                continue
            if(delta_ix<0 or delta_ix>=wnei or delta_iy<0 or delta_iy>=hnei):
                continue
            else:
                te[l,delta_iy,delta_ix,s_iy,s_ix]=1
            #generate te_mask
            #source related
            te_mask[l,:,:,s_iy,s_ix]=1
            #dest related 
            for i in range(-wnei//2,wnei//2+1):
                for j in range(-hnei//2,hnei//2+1):
                    sub_ix=d_ix-i
                    sub_iy=d_iy-j
                    if(sub_ix<0 or sub_ix>=wout or sub_iy<0 or sub_iy>=hout):
                        continue
                    te_mask[l,j+hnei//2,i+wnei//2,sub_iy,sub_ix]=1
    return delta,tx,ty,tw,th,te,te_mask

def draw_results(img,predicts,targets,save_dir,threshold=0.7,name=""):
    pc,px,py,pw,ph,pe=predicts
    tc,tx,ty,tw,th,te,_=targets
    _,_,hin,win=img.shape

    def restore_coor(x,y,w,h):
        grid_size_x=win/wout
        grid_size_y=hin/hout
        grid_x,grid_y=tf.meshgrid(np.arange(wout).astype(np.float32),np.arange(hout).astype(np.float32))
        rx=(x+grid_x)*grid_size_x
        ry=(y+grid_y)*grid_size_y
        rw=w*win
        rh=h*hin
        return rx,ry,rw,rh

    def draw_bbx(img,img_pc,rx,ry,rw,rh,threshold=0.7):
        color=(0,255,0)
        valid_idxs=np.where(img_pc>=threshold,1,0)
        ks,iys,ixs=np.nonzero(valid_idxs)
        for k,iy,ix in zip(ks,iys,ixs):
            x=rx[k][iy][ix]
            y=ry[k][iy][ix]
            w=rw[k][iy][ix]
            h=rh[k][iy][ix]
            img=cv2.circle(img,(int(x),int(y)),radius=2,color=color,thickness=-1)
            img=cv2.rectangle(img,(int(x-w//2),int(y-h//2)),(int(x+w//2),int(y+h//2)),color,1)
        return img

    def draw_edge(img,img_e,rx,ry,rw,rh,threshold=0.7):
        color=(255,0,0)
        valid_idxs=np.where(img_e>=threshold,1,0)
        ls,niys,nixs,iys,ixs=np.nonzero(valid_idxs)
        for l,niy,nix,iy,ix in zip(ls,niys,nixs,iys,ixs):
            s_id=limbs[l][0]-1
            d_id=limbs[l][1]-1
            #get src point
            src_ix=ix
            src_iy=iy
            src_x=rx[s_id][src_iy][src_ix]
            src_y=ry[s_id][src_iy][src_ix]
            #get dst point
            dst_ix=ix-wnei//2+nix
            dst_iy=iy-hnei//2+niy
            if(dst_ix<0 or dst_ix>=wout or dst_iy<0 or dst_iy>=hout):
                continue
            dst_x=rx[d_id][dst_iy][dst_ix]
            dst_y=ry[d_id][dst_iy][dst_ix]
            #draw line
            img=cv2.line(img,(int(src_x),int(src_y)),(int(dst_x),int(dst_y)),color,2)
        return img
    img=np.clip(img*255.0,0,255).astype(np.uint8)
    batch_size,hin,win,_=img.shape
    batch_size,K,hout,wout=px.shape
    batch_size,L,hnei,wnei,hout,wout=pe.shape
    limbs = list(
        zip([2, 9,  10,  2, 12, 13, 2, 3, 4,  2, 6, 7,  2, 1,  1,  15, 16, 19, 19, 19],
            [9, 10, 11, 12, 13, 14, 3, 4, 5,  6, 7, 8,  1, 15, 16, 17, 18, 1 , 17, 18]))
    rtx,rty,rtw,rth=restore_coor(tx,ty,tw,th)
    rpx,rpy,rpw,rph=restore_coor(px,py,pw,ph)
    for b_idx in range(0,batch_size):
        fig=plt.figure(figsize=(8,8))
        #draw original image
        sub_plot=fig.add_subplot(2,2,1)
        sub_plot.set_title("Originl image")
        sub_plot.imshow(img[b_idx].copy())
        #draw predict image
        sub_plot=fig.add_subplot(2,2,2)
        sub_plot.set_title("Predict image")
        img_pd=img[b_idx].copy()
        img_pd=draw_bbx(img_pd,pc[b_idx],rpx[b_idx],rpy[b_idx],rpw[b_idx],rph[b_idx],threshold)
        img_pd=draw_edge(img_pd,pe[b_idx],rpx[b_idx],rpy[b_idx],rpw[b_idx],rph[b_idx],threshold)
        sub_plot.imshow(img_pd)
        #draw ground truth image
        sub_plot=fig.add_subplot(2,2,3)
        sub_plot.set_title("True image")
        img_gt=img[b_idx].copy()
        img_gt=draw_bbx(img_gt,tc[b_idx],rtx[b_idx],rty[b_idx],rtw[b_idx],rth[b_idx],threshold)
        img_gt=draw_edge(img_gt,te[b_idx],rtx[b_idx],rty[b_idx],rtw[b_idx],rth[b_idx],threshold)
        sub_plot.imshow(img_gt)
        #save figure
        plt.savefig(os.path.join(save_dir, '%s_%d.png' % (name, b_idx)), dpi=300)
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
