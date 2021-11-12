import os
import cv2
import functools
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .define import area_ref,area_ref_45
from .define import COCO_SIGMA,COCO_UPRIGHT_POSE,COCO_UPRIGHT_POSE_45
from ..common import regulize_loss, get_meshgrid

def nan2zero(x):
    x=np.where(x!=x,0,x)
    return x

def nan2zero_dict(dict_x):
    for key in dict_x.keys():
        dict_x[key]=nan2zero(dict_x[key])
    return dict_x

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
    #TODO: change mask shape here
    #init fields
    pif_conf=np.full(shape=(n_pos,padded_h,padded_w),fill_value=0.0,dtype=np.float32)
    pif_vec=np.full(shape=(n_pos,2,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    pif_bmin=np.full(shape=(n_pos,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    pif_scale=np.full(shape=(n_pos,padded_h,padded_w),fill_value=np.nan,dtype=np.float32)
    pif_vec_norm=np.full(shape=(n_pos,padded_h,padded_w),fill_value=np.inf,dtype=np.float32)
    #print(f"pif_vec_norm:{pif_vec_norm.shape} pif_conf:{pif_conf.shape} mask:{mask.shape}")
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

def add_gaussian(hr_conf,confs,vecs,sigmas,truncate=1.0,max_value=1.0,neighbor_num=16,debug=False):
    if(debug):
        print()
    field_h,field_w=hr_conf.shape
    for conf,vec,scale in zip(confs,vecs,sigmas):
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
    hr_conf=np.zeros(shape=(field_num,(hout-1)*stride+1,(wout-1)*stride+1))
    for field_idx in range(0,field_num):
        #filter by thresh
        if(debug):
            print(f"\ngenerating hr_conf {field_idx}:")
        thresh_mask=conf_map[field_idx]>thresh
        confs=conf_map[field_idx][thresh_mask]
        vecs=vec_map[field_idx,:,thresh_mask]
        scales=scale_map[field_idx][thresh_mask]
        if(debug):
            print(f"test filed_idx:{field_idx} scale_mean:{np.mean(scales/stride)} scale_var:{np.var(scales/stride)}")
        sigmas=np.maximum(1.0,0.5*scales)
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
    radius=max(np.round(min(image_h,image_w)/300).astype(np.int),1)
    thickness=max(np.round(min(image_h,image_w)/240).astype(np.int),1)
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

def restore_pif_maps(pif_vec_map_batch, pif_scale_map_batch, stride=8):
    hout, wout = pif_vec_map_batch.shape[-2], pif_vec_map_batch.shape[-1]
    mesh_grid = get_meshgrid(mesh_h=hout, mesh_w=wout)
    pif_vec_map_batch = (pif_vec_map_batch + mesh_grid)*stride
    pif_scale_map_batch = pif_scale_map_batch*stride
    return pif_vec_map_batch, pif_scale_map_batch

def restore_paf_maps(paf_src_vec_map_batch, paf_dst_vec_map_batch, paf_src_scale_map_batch, paf_dst_scale_map_batch, stride=8):
    hout, wout = paf_src_vec_map_batch.shape[-2], paf_src_vec_map_batch.shape[-1]
    mesh_grid = get_meshgrid(mesh_h=hout, mesh_w=wout)
    paf_src_vec_map_batch = (paf_src_vec_map_batch + mesh_grid)*stride
    paf_dst_vec_map_batch = (paf_dst_vec_map_batch + mesh_grid)*stride
    paf_src_scale_map_batch = paf_src_scale_map_batch*stride
    paf_dst_scale_map_batch = paf_dst_scale_map_batch*stride
    return paf_src_vec_map_batch, paf_dst_vec_map_batch, paf_src_scale_map_batch, paf_dst_scale_map_batch

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

def pixel_shuffle(x,scale):
    b,c,h,w=x.shape
    new_c=c//(scale**2)
    new_h=h*scale
    new_w=w*scale
    ta=tf.reshape(x,[b,new_c,scale,scale,h,w])
    tb=tf.transpose(ta,[0, 1, 4, 2, 5, 3])
    tc=tf.reshape(tb,[b,new_c,new_h,new_w])
    return tc














