import os
import cv2
import math
import logging
import functools
import numpy as np
import tensorflow as tf
from .define import COCO_SIGMA,COCO_UPRIGHT_POSE,COCO_UPRIGHT_POSE_45
from .define import area_ref,area_ref_45

@functools.lru_cache(maxsize=64)
def get_patch_meshgrid(patch_size):
    x_range=np.linspace(start=(patch_size-1)/2,stop=-(patch_size-1)/2,num=patch_size)
    y_range=np.linspace(start=(patch_size-1)/2,stop=-(patch_size-1)/2,num=patch_size)
    mesh_x,mesh_y=np.meshgrid(x_range,y_range)
    patch_grid=np.stack([mesh_x,mesh_y])
    return patch_grid

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
    ref_pose=np.copy(COCO_UPRIGHT_POSE)
    ref_pose_45=np.copy(COCO_UPRIGHT_POSE_45)
    visible=np.logical_not(np.logical_and(keypoints[:,0]<0,keypoints[:,1]<0))
    #calculate visible area
    area_vis=(np.max(keypoints[visible, 0]) - np.min(keypoints[visible, 0]))*\
            (np.max(keypoints[visible, 1]) - np.min(keypoints[visible, 1]))
    #calculate reference visible area
    area_ref_vis=(np.max(ref_pose[visible, 0]) - np.min(ref_pose[visible, 0]))*\
            (np.max(ref_pose[visible, 1]) - np.min(ref_pose[visible, 1]))
    factor_ref_vis=area_ref/area_ref_vis if area_ref_vis>0.1 else np.inf
    #calculate reference 45 rotated visible area
    area_ref_45_vis=(np.max(ref_pose_45[visible, 0]) - np.min(ref_pose_45[visible, 0]))*\
            (np.max(ref_pose_45[visible, 1]) - np.min(ref_pose_45[visible, 1]))
    factor_ref_45_vis=area_ref_45/area_ref_45_vis if area_ref_45_vis>0.1 else np.inf
    #calculate scale-factor
    if(factor_ref_vis==np.inf and factor_ref_45_vis==np.inf):
        factor=1.0
    else:
        factor=np.sqrt(min(factor_ref_vis,factor_ref_45_vis))
    factor=min(factor,5.0)
    scale=np.sqrt(area_vis)*factor
    #in original pifpaf, scale<0.1 should be set to nan
    scale=min(scale,0.1)
    return scale

def get_pifmap(annos, mask, height, width, hout, wout, parts, limbs,dist_thresh=1.0,patch_size=4,padding=10, data_format="channels_first"):
    n_pos,n_limbs=len(parts),len(limbs)
    padded_h,padded_w=hout+2*padding,wout+2*padding
    #init maps
    pif_conf=np.zeros(shape=(n_pos,padded_h,padded_w))
    pif_vec=np.zeros(shape=(n_pos,2,padded_h,padded_w))
    pif_scale=np.zeros(shape=(n_pos,padded_h,padded_w))
    pif_vec_norm=np.zeros(shape=(n_pos,padded_h,padded_w))
    inv_mask=1-mask
    pif_vec_norm[:,inv_mask==1]=dist_thresh
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
            kpt_scale=min(anno_scale*COCO_SIGMA[part_idx],np.min(max_r)*0.25)
            pif_conf,pif_vec,pif_scale,pif_vec_norm=put_pifmap(pif_conf,pif_vec,pif_scale,pif_vec_norm,\
                part_idx,kpt,kpt_scale=kpt_scale,dist_thresh=dist_thresh,patch_size=patch_size,padding=padding)
    #get field without padding (TODO: valid area?)
    pif_conf=pif_conf[:,padding:-padding,padding:-padding]
    pif_vec=pif_vec[:,:,padding:-padding,padding:-padding]
    pif_scale=pif_scale[:,padding:-padding,padding:-padding]
    return pif_conf,pif_vec,pif_scale

def put_pifmap(pif_conf,pif_vec,pif_scale,pif_vec_norm,part_idx,kpt,kpt_scale,dist_thresh=1.0,patch_size=4,padding=10):
    padded_h,padded_w=pif_conf.shape[1],pif_conf.shape[2]
    #calculate patch grid coordinate range in padded map
    patch_offset=(patch_size-1)/2
    left_top=np.round(kpt-patch_offset+padding).astype(np.int)
    min_x,min_y=left_top[0],left_top[1]
    max_x,max_y=min_x+patch_size,min_y+patch_size
    if(min_x<0 or min_x>=padded_w or max_y<0 or max_y>=padded_h):
        return pif_conf,pif_vec,pif_scale,pif_vec_norm
    #calculate mesh center to kpt offset
    patch_center_offset=kpt-(left_top+patch_offset-padding)
    #calculate mesh grid to mesh center offset
    patch_meshgrid=get_patch_meshgrid(patch_size)
    #calculate mesh grid to kpt offset
    patch_grid_offset=patch_meshgrid+patch_center_offset[:,np.newaxis,np.newaxis]
    patch_grid_offset_norm=np.linalg.norm(patch_grid_offset,axis=0)
    grid_mask=patch_grid_offset_norm<pif_vec_norm[part_idx,min_y:max_y,min_x:max_x]
    
    #update pif_vec_norm (to closet distance)
    pif_vec_norm[part_idx,min_y:max_y,min_x:max_x][grid_mask]=patch_grid_offset_norm[grid_mask]
    #update pif_conf (to 1.0 where less than dist_trhsh)
    pif_conf[part_idx,min_y:max_y,min_x:max_x][grid_mask]=1.0
    #update pif_vec (to patch_grid to kpt offset)
    pif_vec[part_idx,:,min_y:max_y,min_x:max_x][grid_mask]=patch_grid_offset[grid_mask]
    #update pif_scale (to kpt scale)
    pif_scale[part_idx,min_y:max_y,min_x:max_x][grid_mask]=kpt_scale
    return pif_conf,pif_vec,pif_scale,pif_vec_norm


    
    

            


