import os
import cv2
import math
import logging
import numpy as np
import tensorflow as tf
from .define import COCO_SIGMA,COCO_UPRIGHT_POSE

def get_max_r(kpt,other_kpts):
    kpt=np.array(kpt)
    other_kpts=np.array(other_kpts)
    min_dist=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
    dif=other_kpts-kpt[np.newaxis,:]
    mask=np.zeros(shape=(other_kpts.shape[0]))
    mask[dif[:,0]<0.0]+=1
    mask[dif[:,1]<0.0]+=2
    for quadrant in range(0,4):
        if(not np.any(mask==quadrant)):
            continue
        min_dist[quadrant]=np.min(np.linalg.norm(dif[mask==quadrant],axis=1))
    return min_dist

def get_scale(keypoints):


def get_pifmap(annos, mask, height, width, hout, wout, parts, limbs, padding=10, data_format="channels_first"):
    n_pos,n_limbs=len(parts),len(limbs)
    pad_h,pad_w=hout+2*padding,wout+2*padding
    #init maps
    pif_conf=np.zeros(shape=(n_pos,pad_h,pad_w))
    pif_vec=np.zeros(shape=(n_pos,2,pad_h,pad_w))
    pif_scale=np.zeros(shape=(n_pos,pad_h,pad_w))
    pif_vec_norm=np.zeros(shape=(n_pos,pad_h,pad_w))
    inv_mask=1-mask
    pif_vec_norm[:,inv_mask==1]=1.0
    #generate maps
    for anno_id,anno in enumerate(annos):
        other_annos=[other_anno for other_id,other_anno in enumerate(annos) if other_id!=anno_id]
        anno_scale=get_scale(anno)
        for part_idx,kpt in enumerate(anno):
            if(kpt[0]<0 or kpt[0]>width or kpt[1]<0 or kpt[1]>height):
                continue
            other_kpts=[]
            for other_anno in other_annos:
                other_kpt=other_anno[part_idx]
                if(other_kpt[0]<0 or other_kpt[0]>width or other_kpt[1]<0 or other_kpt[1]>height):
                    continue
                other_kpts.append(other_kpt)
            max_r=get_max_r(kpt,other_kpts)
            joint_scale=min(anno_scale*part_sigma[part_idx],np.min(max_r)*0.25)
            pif_conf,pif_vec,pif_scale,pif_vec_norm=put_pifmap(pif_conf,pif_vec,pif_scale,pif_vec_norm,part_idx,kpt,pther_kpts)

def put_pifmap(pif_conf,pif_vec,pif_scale,pif_vec_norm,part_idx,kpt,oth):
    scale
            


