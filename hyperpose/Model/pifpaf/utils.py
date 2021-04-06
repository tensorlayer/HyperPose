import os
import cv2
import math
import logging
import functools
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .define import area_ref,area_ref_45
from .define import COCO_SIGMA,COCO_UPRIGHT_POSE,COCO_UPRIGHT_POSE_45

def nan2zero(x):
    x=np.where(x!=x,0,x)
    return x

def maps_to_numpy(maps):
    ret_maps=[]
    for m_idx,m in enumerate(maps):
        ret_maps.append(m.numpy())
    return ret_maps

@functools.lru_cache(maxsize=64)
def get_patch_meshgrid(patch_size):
    x_range=np.linspace(start=(patch_size-1)/2,stop=-(patch_size-1)/2,num=patch_size)
    y_range=np.linspace(start=(patch_size-1)/2,stop=-(patch_size-1)/2,num=patch_size)
    mesh_x,mesh_y=np.meshgrid(x_range,y_range)
    patch_grid=np.stack([mesh_x,mesh_y])
    return patch_grid

def get_max_r(kpt,other_kpts):
    min_dist=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
    if(len(other_kpts)!=0):
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
    keypoints=np.array(keypoints)
    ref_pose=np.copy(COCO_UPRIGHT_POSE)
    ref_pose_45=np.copy(COCO_UPRIGHT_POSE_45)
    visible=np.logical_not(np.logical_and(keypoints[:,0]<0,keypoints[:,1]<0))
    if(np.sum(visible)<=3):
        return None
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
    #print(f"test label scale area_vis:{area_vis} area_ref_vis:{area_ref_vis} area_ref_45_vis:{area_ref_45_vis} "+\
    #    f"factor_ref_vis:{factor_ref_vis} factor_ref_45_vis:{factor_ref_45_vis} factor:{factor} scale:{scale}\n")
    #in original pifpaf, scale<0.1 should be set to nan
    scale=max(scale,0.1)
    return scale

def get_pifmap(annos, mask, height, width, hout, wout, parts, limbs,bmin=0.1,dist_thresh=1.0,patch_size=4,padding=10, data_format="channels_first"):
    stride=height/hout
    strided_bmin=bmin/stride
    n_pos,n_limbs=len(parts),len(limbs)
    padded_h,padded_w=hout+2*padding,wout+2*padding
    #init fields
    pif_conf=np.full(shape=(n_pos,padded_h,padded_w),fill_value=0.0,dtype=np.float32)
    pif_vec=np.full(shape=(n_pos,2,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    pif_bmin=np.full(shape=(n_pos,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    pif_scale=np.full(shape=(n_pos,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    pif_vec_norm=np.full(shape=(n_pos,padded_h,padded_w),fill_value=np.inf,dtype=np.float32)
    pif_vec_norm[:,padding:-padding,padding:-padding][:,mask==0]=dist_thresh
    pif_conf[:,padding:-padding,padding:-padding][:,mask==0]=np.nan
    #generate fields
    for anno_id,anno in enumerate(annos):
        anno_scale=get_scale(np.array(anno)/stride)
        if(anno_scale==None):
            continue
        for part_idx,kpt in enumerate(anno):
            if(kpt[0]<0 or kpt[0]>width or kpt[1]<0 or kpt[1]>height):
                continue
            #calculate scale
            kpt=np.array(kpt)/stride
            kpt_scale=anno_scale*COCO_SIGMA[part_idx]
            #print(f"test pif scale:")
            #print(f"kpt_idx:{part_idx} kpt_scale:{kpt_scale}")
            #generate pif_maps for single point
            pif_maps=[pif_conf,pif_vec,pif_bmin,pif_scale,pif_vec_norm]
            pif_conf,pif_vec,pif_bmin,pif_scale,pif_vec_norm=put_pifmap(pif_maps,part_idx,kpt,\
                kpt_scale=kpt_scale,strided_bmin=strided_bmin,dist_thresh=dist_thresh,patch_size=patch_size,padding=padding)
    #get field without padding (TODO: valid area?)
    pif_conf=pif_conf[:,padding:-padding,padding:-padding]
    pif_vec=pif_vec[:,:,padding:-padding,padding:-padding]
    pif_bmin=pif_bmin[:,padding:-padding,padding:-padding]
    pif_scale=pif_scale[:,padding:-padding,padding:-padding]
    return pif_conf,pif_vec,pif_bmin,pif_scale

def put_pifmap(pif_maps,part_idx,kpt,kpt_scale,strided_bmin=0.0125,dist_thresh=1.0,patch_size=4,padding=10):
    pif_conf,pif_vec,pif_bmin,pif_scale,pif_vec_norm=pif_maps
    padded_h,padded_w=pif_conf.shape[1],pif_conf.shape[2]
    #calculate patch grid coordinate range in padded map
    patch_offset=(patch_size-1)/2
    left_top=np.round(kpt-patch_offset+padding).astype(np.int)
    min_x,min_y=left_top[0],left_top[1]
    max_x,max_y=min_x+patch_size,min_y+patch_size
    if(min_x<0 or min_x>=padded_w or max_y<0 or max_y>=padded_h):
        return pif_conf,pif_vec,pif_bmin,pif_scale,pif_vec_norm
    #calculate mesh center to kpt offset
    patch_center_offset=kpt-(left_top+patch_offset-padding)
    #calculate mesh grid to mesh center offset
    patch_meshgrid=get_patch_meshgrid(patch_size)
    #calculate mesh grid to kpt offset
    patch_grid_offset=patch_meshgrid+patch_center_offset[:,np.newaxis,np.newaxis]
    patch_grid_offset_norm=np.linalg.norm(patch_grid_offset,axis=0)
    #calculate mash mask acordding to the distance to the keypoints
    grid_mask=patch_grid_offset_norm<pif_vec_norm[part_idx,min_y:max_y,min_x:max_x]
    
    #update pif_vec_norm (to closet distance)
    pif_vec_norm[part_idx,min_y:max_y,min_x:max_x][grid_mask]=patch_grid_offset_norm[grid_mask]
    #update pif_conf (to 1.0 where less than dist_trhsh)
    pif_conf[part_idx,min_y:max_y,min_x:max_x][grid_mask]=1.0
    #update pif_vec (to patch_grid to kpt offset)
    pif_vec[part_idx,:,min_y:max_y,min_x:max_x][:,grid_mask]=patch_grid_offset[:,grid_mask]
    #update pif_bmin
    pif_bmin[part_idx,min_y:max_y,min_x:max_x][grid_mask]=strided_bmin
    #update pif_scale (to kpt scale)
    pif_scale[part_idx,min_y:max_y,min_x:max_x][grid_mask]=kpt_scale
    return pif_conf,pif_vec,pif_bmin,pif_scale,pif_vec_norm

def get_pafmap(annos,mask,height, width, hout, wout, parts, limbs,bmin=0.1,dist_thresh=1.0,patch_size=3,padding=10, data_format="channels_first"):
    stride=height/hout
    strided_bmin=bmin/stride
    n_pos,n_limbs=len(parts),len(limbs)
    padded_h,padded_w=hout+2*padding,wout+2*padding
    #init fields
    paf_conf=np.full(shape=(n_limbs,padded_h,padded_w),fill_value=0.0,dtype=np.float32)
    paf_src_vec=np.full(shape=(n_limbs,2,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    paf_dst_vec=np.full(shape=(n_limbs,2,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    paf_src_bmin=np.full(shape=(n_limbs,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    paf_dst_bmin=np.full(shape=(n_limbs,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    paf_src_scale=np.full(shape=(n_limbs,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    paf_dst_scale=np.full(shape=(n_limbs,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    paf_vec_norm=np.full(shape=(n_limbs,padded_h,padded_w),fill_value=np.inf,dtype=np.float32)
    paf_vec_norm[:,padding:-padding,padding:-padding][:,mask==0]=1.0
    paf_conf[:,padding:-padding,padding:-padding][:,mask==0]=np.nan
    #generate fields
    for anno_id,anno in enumerate(annos):
        anno_scale=get_scale(np.array(anno)/stride)
        if(anno_scale==None):
            continue
        for limb_idx,(src_idx,dst_idx) in enumerate(limbs):
            src_kpt=np.array(anno[src_idx])/stride
            dst_kpt=np.array(anno[dst_idx])/stride
            out_of_field_src=(src_kpt[0]<0 or src_kpt[0]>=wout or src_kpt[1]<0 or src_kpt[1]>=hout)
            out_of_field_dst=(dst_kpt[0]<0 or dst_kpt[1]>=wout or dst_kpt[1]<0 or dst_kpt[1]>=hout)
            if(out_of_field_src or out_of_field_dst):
                continue
            #calculate src scale
            src_scale=anno_scale*COCO_SIGMA[src_idx]
            #calculate dst scale
            dst_scale=anno_scale*COCO_SIGMA[dst_idx]
            #print(f"test paf scale:")
            #print(f"src_kpt_idx:{src_idx} src_kpt_scale:{src_scale}")
            #print(f"dst_kpt_idx:{dst_idx} dst_kpt_scale:{dst_scale}")
            #generate paf_maps for single point
            paf_maps=[paf_conf,paf_src_vec,paf_dst_vec,paf_src_bmin,paf_dst_bmin,paf_src_scale,paf_dst_scale,paf_vec_norm]
            paf_conf,paf_src_vec,paf_dst_vec,paf_src_bmin,paf_dst_bmin,paf_src_scale,paf_dst_scale,paf_vec_norm=put_pafmap(paf_maps,limb_idx,src_kpt,src_scale,dst_kpt,dst_scale,\
                strided_bmin=strided_bmin,padding=padding,patch_size=patch_size,data_format=data_format)
    #get field without padding (TODO: valid area?)
    paf_conf=paf_conf[:,padding:-padding,padding:-padding]
    paf_src_vec=paf_src_vec[:,:,padding:-padding,padding:-padding]
    paf_dst_vec=paf_dst_vec[:,:,padding:-padding,padding:-padding]
    paf_src_bmin=paf_src_bmin[:,padding:-padding,padding:-padding]
    paf_dst_bmin=paf_dst_bmin[:,padding:-padding,padding:-padding]
    paf_src_scale=paf_src_scale[:,padding:-padding,padding:-padding]
    paf_dst_scale=paf_dst_scale[:,padding:-padding,padding:-padding]
    return paf_conf,paf_src_vec,paf_dst_vec,paf_src_bmin,paf_dst_bmin,paf_src_scale,paf_dst_scale

def put_pafmap(paf_maps,limb_idx,src_kpt,src_scale,dst_kpt,dst_scale,patch_size=3,strided_bmin=0.0125,padding=10,data_format="channels_first"):
    paf_conf,paf_src_vec,paf_dst_vec,paf_src_bmin,paf_dst_bmin,paf_src_scale,paf_dst_scale,paf_vec_norm=paf_maps
    padded_h,padded_w=paf_conf.shape[1],paf_conf.shape[2]
    patch_offset=(patch_size-1)/2
    limb_vec=dst_kpt-src_kpt
    limb_vec_norm=np.linalg.norm(limb_vec)
    meshgrid_offsets=np.stack(np.meshgrid(
            np.linspace(-0.5 * (patch_size - 1), 0.5 * (patch_size - 1), patch_size),
            np.linspace(-0.5 * (patch_size - 1), 0.5 * (patch_size - 1), patch_size),
        ), axis=-1).reshape(-1, 2)
    #split the limb vec into line segmentations to fill the field
    sample_num=max(2,int(np.ceil(limb_vec_norm)))
    fmargin=(patch_size/2)/(limb_vec_norm+np.spacing(1))
    fmargin=np.clip(fmargin,0.25,0.4)
    frange=np.linspace(fmargin,1.0-fmargin,num=sample_num)
    filled_points=set()
    for lmbda in frange:
        for meshgrid_offset in meshgrid_offsets:
            mesh_x,mesh_y=np.round(src_kpt+lmbda*limb_vec+meshgrid_offset).astype(np.int)+padding
            if(mesh_x<0 or mesh_x>=padded_w or mesh_y<0 or mesh_y>=padded_h):
                continue
            #check for repeatly filling the same point
            mesh_coordinate=(int(mesh_x),int(mesh_y))
            if(mesh_coordinate in filled_points):
                continue
            filled_points.add(mesh_coordinate)
            offset=np.array([mesh_x,mesh_y])-padding-src_kpt
            distline=np.fabs(limb_vec[1]*offset[0]-limb_vec[0]*offset[1])/(limb_vec_norm+0.01)
            if(distline<paf_vec_norm[limb_idx,mesh_y,mesh_x]):
                #update paf_vec_norm
                paf_vec_norm[limb_idx,mesh_y,mesh_x]=distline
                #update paf_conf
                paf_conf[limb_idx,mesh_y,mesh_x]=1.0
                #update paf_src_vec
                paf_src_vec[limb_idx,:,mesh_y,mesh_x]=src_kpt-(np.array([mesh_x,mesh_y])-padding)
                #update paf_dst_vec
                paf_dst_vec[limb_idx,:,mesh_y,mesh_x]=dst_kpt-(np.array([mesh_x,mesh_y])-padding)
                #update paf_src_bmin
                paf_src_bmin[limb_idx,mesh_y,mesh_x]=strided_bmin
                #update paf_dst_bmin
                paf_dst_bmin[limb_idx,mesh_y,mesh_x]=strided_bmin
                #update src_scale
                paf_src_scale[limb_idx,mesh_y,mesh_x]=src_scale
                #update dst_scale
                paf_dst_scale[limb_idx,mesh_y,mesh_x]=dst_scale
    return paf_conf,paf_src_vec,paf_dst_vec,paf_src_bmin,paf_dst_bmin,paf_src_scale,paf_dst_scale,paf_vec_norm

def add_gaussian(hr_conf,confs,vecs,scales,truncate=1.0,max_value=1.0,neighbor_num=9,debug=False):
    if(debug):
        print()
    field_h,field_w=hr_conf.shape
    for conf,vec,scale in zip(confs,vecs,scales):
        x,y=vec
        #calculate mesh range
        min_x=np.clip(x-truncate*scale,0,field_w-1).astype(np.int)
        max_x=np.clip(x+truncate*scale+1,min_x+1,field_w).astype(np.int)
        min_y=np.clip(y-truncate*scale,0,field_h-1).astype(np.int)
        max_y=np.clip(y+truncate*scale+1,min_y+1,field_h).astype(np.int)
        #calculate mesh grid
        x_range=np.linspace(start=min_x,stop=max_x-1,num=max_x-min_x)
        y_range=np.linspace(start=min_y,stop=max_y-1,num=max_y-min_y)
        mesh_x,mesh_y=np.meshgrid(x_range,y_range)
        #calculate gaussian heatmap according to the mesh grid distance
        mesh_dist=(mesh_x-x)**2+(mesh_y-y)**2
        mesh_mask=np.where(mesh_dist<=((scale*truncate)**2),True,False)
        mesh_update_conf=conf*np.exp(-0.5*mesh_dist/(scale**2))
        #adjust heatmap score of the center point
        center_x,center_y=np.round(x).astype(np.int),np.round(y).astype(np.int)
        if(center_x>=min_x and center_x<max_x and center_y>=min_y and center_y<max_y):
            mesh_update_conf[center_y-min_y,center_x-min_x]=conf
        if(debug):
            print(f"test hr_conf.shape:{hr_conf.shape} scale:{scale} x:{x} y:{y} min_x:{min_x} max_x:{max_x} min_y:{min_y} max_y:{max_y} conf:{conf} scale:{scale}")
            print(f"center_x:{center_x} center_y:{center_y} \nmesh_update_conf:\n{mesh_update_conf}")
        #update heatmap according to distance mask
        #TODO: original code divide add by neighbor_num, this will result in larger target get higher score
        #so judge whether should divide this by scale_size
        hr_conf[min_y:max_y,min_x:max_x][mesh_mask]+=mesh_update_conf[mesh_mask]/neighbor_num
    hr_conf=np.clip(hr_conf,0.0,max_value)
    return hr_conf

def get_hr_conf(conf_map,vec_map,scale_map,stride=8,thresh=0.1,debug=False):
    #shape
    #conf [field_num,hout,wout]
    #vec [field_num,2,hout,wout]
    #scale [field_num,hout,wout]
    field_num,hout,wout=conf_map.shape
    hr_conf=np.zeros(shape=(field_num,hout*stride,wout*stride))
    for field_idx in range(0,field_num):
        #filter by thresh
        if(debug):
            print(f"\ngenerating hr_conf {field_idx}:")
        thresh_mask=conf_map[field_idx]>thresh
        confs=conf_map[field_idx][thresh_mask]
        vecs=vec_map[field_idx,:,thresh_mask]
        scales=np.maximum(1.0,0.75*scale_map[field_idx][thresh_mask])
        hr_conf[field_idx]=add_gaussian(hr_conf[field_idx],confs,vecs,scales,debug=debug)
    return hr_conf

def get_arrow_map(array_map,conf_map,src_vec_map,dst_vec_map,thresh=0.1,src_color=(255,255,0),dst_color=(0,0,255),debug=False):
    #make integer indexes
    def toidx(x):
        return np.round(x).astype(np.int)
    #shape conf:[field,h,w]
    #shape vec:[field,2,h,w]
    #shape array:[h,w,3]
    grid_center_color=(165,42,42)
    src_center_color=(179,238,58)
    dst_center_color=(30,144,255)
    image_h,image_w,_=array_map.shape
    stride=image_h/conf_map.shape[1]
    radius=np.round(min(image_h,image_w)/300).astype(np.int)
    thickness=np.round(min(image_h,image_w)/240).astype(np.int)
    mask=conf_map>thresh
    fields,grid_ys,grid_xs=np.where(mask)
    for field,grid_y,grid_x in zip(fields,grid_ys,grid_xs):
        src_x,src_y=toidx(src_vec_map[field,:,grid_y,grid_x])
        dst_x,dst_y=toidx(dst_vec_map[field,:,grid_y,grid_x])
        grid_y,grid_x=toidx(grid_y*stride),toidx(grid_x*stride)
        array_map=cv2.circle(array_map,(grid_x,grid_y),radius=radius,color=grid_center_color,thickness=thickness)
        if(debug):
            print(f"test get_arrow_map image_h:{image_h} image_w:{image_w} field:{field} gird_x:{grid_x} grid_y:{grid_y} src_x:{src_x} src_y:{src_y} dst_x:{dst_x} dst_y:{dst_y}")
        if(src_x>=0 and src_x<image_w and src_y>=0 and src_y<image_h):
            array_map=cv2.circle(array_map,(src_x,src_y),radius=radius,color=src_center_color,thickness=thickness)
            array_map=cv2.line(array_map,(grid_x,grid_y),(src_x,src_y),color=src_color,thickness=thickness)
        if(dst_x>=0 and dst_x<image_w and dst_y>=0 and dst_y<image_h):
            array_map=cv2.circle(array_map,(dst_x,dst_y),radius=radius,color=dst_center_color,thickness=thickness)
            array_map=cv2.line(array_map,(grid_x,grid_y),(dst_x,dst_y),color=dst_color,thickness=thickness)
    return array_map

def draw_result(images,pd_pif_maps,pd_paf_maps,gt_pif_maps,gt_paf_maps,masks,parts,limbs,stride=8,thresh_pif=0.1,thresh_paf=0.1,\
    save_dir="./save_dir",name="default"):
    #shape
    #conf [batch_size,field_num,hout,wout]
    #vec [batch_size,field_num,2,hout,wout]
    #scale [batch_size,field_num,hout,wout]
    #decode pif_maps
    pd_pif_conf,pd_pif_vec,_,pd_pif_scale=pd_pif_maps
    gt_pif_conf,gt_pif_vec,_,gt_pif_scale=gt_pif_maps
    #decode paf_maps
    pd_paf_conf,pd_paf_src_vec,pd_paf_dst_vec,_,_,_,_=pd_paf_maps
    gt_paf_conf,gt_paf_src_vec,gt_paf_dst_vec,_,_,_,_=gt_paf_maps
    #restore conf_maps
    pd_pif_conf=tf.nn.sigmoid(pd_pif_conf).numpy()
    pd_paf_conf=tf.nn.sigmoid(pd_paf_conf).numpy()
    pd_pif_scale=tf.math.softplus(pd_pif_scale)
    #restore nan in gt_maps
    gt_pif_conf=nan2zero(gt_pif_conf)
    gt_pif_vec=nan2zero(gt_pif_vec)
    gt_pif_scale=nan2zero(gt_pif_scale)
    gt_paf_conf=nan2zero(gt_paf_conf)
    gt_paf_src_vec=nan2zero(gt_paf_src_vec)
    gt_paf_dst_vec=nan2zero(gt_paf_dst_vec)
    #restore vector maps
    _,_,hout,wout=pd_pif_conf.shape
    x_range=np.linspace(start=0,stop=wout-1,num=wout)
    y_range=np.linspace(start=0,stop=hout-1,num=hout)
    mesh_x,mesh_y=np.meshgrid(x_range,y_range)
    mesh_grid=np.stack([mesh_x,mesh_y])
    pd_pif_scale=pd_pif_scale*stride
    gt_pif_scale=gt_pif_scale*stride
    pd_pif_vec=(pd_pif_vec+mesh_grid)*stride
    gt_pif_vec=(gt_pif_vec+mesh_grid)*stride
    pd_paf_src_vec=(pd_paf_src_vec+mesh_grid)*stride
    pd_paf_dst_vec=(pd_paf_dst_vec+mesh_grid)*stride
    gt_paf_src_vec=(gt_paf_src_vec+mesh_grid)*stride
    gt_paf_dst_vec=(gt_paf_dst_vec+mesh_grid)*stride
    #draw
    os.makedirs(save_dir,exist_ok=True)
    batch_size=pd_paf_conf.shape[0]
    for batch_idx in range(0,batch_size):
        #image and mask
        image_show=images[batch_idx].transpose([1,2,0])
        mask_show=masks[batch_idx][0]

        #draw pif maps
        #pif_conf_map
        pd_pif_conf_show=np.amax(pd_pif_conf[batch_idx],axis=0)
        gt_pif_conf_show=np.amax(gt_pif_conf[batch_idx],axis=0)
        #pif_hr_conf_map
        pd_pif_hr_conf=get_hr_conf(pd_pif_conf[batch_idx],pd_pif_vec[batch_idx],pd_pif_scale[batch_idx],stride=stride,thresh=thresh_pif)
        pd_pif_hr_conf_show=np.amax(pd_pif_hr_conf,axis=0)
        gt_pif_hr_conf=get_hr_conf(gt_pif_conf[batch_idx],gt_pif_vec[batch_idx],gt_pif_scale[batch_idx],stride=stride,thresh=thresh_pif)
        gt_pif_hr_conf_show=np.amax(gt_pif_hr_conf,axis=0)
        #plt draw
        fig=plt.figure(figsize=(8,8))
        #show image
        a=fig.add_subplot(2,3,1)
        a.set_title("image")
        plt.imshow(image_show)
        #show gt_pif_conf
        a=fig.add_subplot(2,3,2)
        a.set_title("gt_pif_conf")
        plt.imshow(gt_pif_conf_show,alpha=0.8)
        plt.colorbar()
        #show gt_pif_hr_conf
        a=fig.add_subplot(2,3,3)
        a.set_title("gt_pif_hr_conf")
        plt.imshow(gt_pif_hr_conf_show,alpha=0.8)
        plt.colorbar()
        #show mask
        a=fig.add_subplot(2,3,4)
        a.set_title("mask")
        plt.imshow(mask_show)
        plt.colorbar()
        #show pd_pif_conf
        a=fig.add_subplot(2,3,5)
        a.set_title("pd_pif_conf")
        plt.imshow(pd_pif_conf_show,alpha=0.8)
        plt.colorbar()
        #show pd_pif_hr_conf
        a=fig.add_subplot(2,3,6)
        a.set_title("pd_pif_hr_conf")
        plt.imshow(pd_pif_hr_conf_show,alpha=0.8)
        plt.colorbar()
        #save drawn figures
        plt.savefig(os.path.join(save_dir,f"{name}_{batch_idx}_pif.png"),dpi=400)
        plt.close()

        #draw paf maps
        #paf_conf_map
        pd_paf_conf_show=np.amax(pd_paf_conf[batch_idx],axis=0)
        gt_paf_conf_show=np.amax(gt_paf_conf[batch_idx],axis=0)
        #paf_vec_map
        #pd_paf_vec_map
        pd_paf_vec_show=np.zeros(shape=(hout*stride,wout*stride,3)).astype(np.int8)
        pd_paf_vec_show=get_arrow_map(pd_paf_vec_show,pd_paf_conf[batch_idx],pd_paf_src_vec[batch_idx],pd_paf_dst_vec[batch_idx],thresh_paf)
        #gt_paf_vec_map
        gt_paf_vec_show=np.zeros(shape=(hout*stride,wout*stride,3)).astype(np.int8)
        gt_paf_vec_show=get_arrow_map(gt_paf_vec_show,gt_paf_conf[batch_idx],gt_paf_src_vec[batch_idx],gt_paf_dst_vec[batch_idx],thresh_paf,debug=False)
        #plt draw
        fig=plt.figure(figsize=(8,8))
        #show image
        a=fig.add_subplot(2,3,1)
        a.set_title("image")
        plt.imshow(image_show)
        #show gt_paf_conf
        a=fig.add_subplot(2,3,2)
        a.set_title("gt_paf_conf")
        plt.imshow(gt_paf_conf_show,alpha=0.8)
        plt.colorbar()
        #show gt_paf_vec_conf
        a=fig.add_subplot(2,3,3)
        a.set_title("gt_paf_vec_conf")
        plt.imshow(gt_paf_vec_show,alpha=0.8)
        plt.colorbar()
        #show mask
        a=fig.add_subplot(2,3,4)
        a.set_title("mask")
        plt.imshow(mask_show)
        plt.colorbar()
        #show pd_paf_conf
        a=fig.add_subplot(2,3,5)
        a.set_title("pd_paf_conf")
        plt.imshow(pd_paf_conf_show,alpha=0.8)
        plt.colorbar()
        #show pd_paf_vec_conf
        a=fig.add_subplot(2,3,6)
        a.set_title("pd_paf_vec_show")
        plt.imshow(pd_paf_vec_show,alpha=0.8)
        plt.colorbar()
        #save drawn figures
        plt.savefig(os.path.join(save_dir,f"{name}_{batch_idx}_paf.png"),dpi=400)
        plt.close()

from ..common import DATA
from .define import CocoPart,CocoLimb,CocoColor
from .define import MpiiPart,MpiiLimb,MpiiColor

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

        












