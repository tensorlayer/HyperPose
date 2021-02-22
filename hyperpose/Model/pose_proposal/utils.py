import os
import cv2
import numpy as np
import tensorflow as tf
from tensorlayer import logging
from tensorlayer.files.utils import (del_file, folder_exists, maybe_download_and_extract)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from distutils.dir_util import mkpath
from scipy.spatial.distance import cdist
from pycocotools.coco import COCO, maskUtils

def get_pose_proposals(kpts_list,bbxs,hin,win,hout,wout,hnei,wnei,parts,limbs,img_mask=None,data_format="channels_first"):
    K,L=len(parts),len(limbs)
    grid_x=win/wout
    grid_y=hin/hout
    delta=np.zeros(shape=(K,hout,wout))
    tx=np.zeros(shape=(K,hout,wout))
    ty=np.zeros(shape=(K,hout,wout))
    tw=np.zeros(shape=(K,hout,wout))
    th=np.zeros(shape=(K,hout,wout))
    te=np.zeros(shape=(L,hnei,wnei,hout,wout))
    te_mask=np.zeros(shape=(L,hnei,wnei,hout,wout))
    aux_delta=np.zeros(shape=(hout+hnei-1,wout+wnei-1,K,2))
    #generate pose proposals for each labels person
    for human_idx,(kpts,bbx) in enumerate(zip(kpts_list,bbxs)):
        #change the background keypoint to the added human instance keypoint
        _,_,ins_w,ins_h=bbx
        part_size=int(max(ins_w,ins_h)/8)
        instance_size=int(max(ins_w,ins_h)/4)
        for k,kpt in enumerate(kpts):
            x,y=kpt[0],kpt[1]
            if(x<0 or y<0 or x>=win or y>=hin):
                continue
            if(img_mask.all()!=None):
                #joints are masked
                if(img_mask[int(x),int(y)]==0):
                    continue
            cx,cy=x/grid_x,y/grid_y
            ix,iy=int(cx),int(cy)
            delta[k,iy,ix]=1
            aux_delta[iy+(hnei//2),ix+(wnei//2),k,0]=1
            aux_delta[iy+(hnei//2),ix+(wnei//2),k,1]=human_idx
            tx[k,iy,ix]=cx-ix
            ty[k,iy,ix]=cy-iy
            if(k==parts.Instance.value):
                size=instance_size
            else:
                size=part_size
            tw[k,iy,ix]=size/win
            th[k,iy,ix]=size/hin
                
    #generate te and mask
    np_limbs=np.array(limbs)
    limbs_start=np_limbs[:,0]
    limbs_end=np_limbs[:,1]
    for iy in range(0,hout):
        for ix in range(0,wout):
            start=aux_delta[iy+(hnei//2),ix+(wnei//2),limbs_start,:]
            end=aux_delta[iy:iy+(hnei//2)*2+1,ix:ix+(wnei//2)*2+1,limbs_end,:]
            te_mask[:,:,:,iy,ix]=(np.maximum(start[:,0],end[:,:,:,0])).transpose(2,0,1)
            condition=np.logical_and((start[:,0]*end[:,:,:,0]==1),start[:,1]==end[:,:,:,1])
            te[:,:,:,iy,ix]=(np.where(condition,1,0)).transpose(2,0,1)
    
    if(data_format=="channels_last"):
        delta=np.transpose(delta,[1,2,0])
        tx=np.transpose(tx,[1,2,0])
        ty=np.transpose(ty,[1,2,0])
        tw=np.transpose(tw,[1,2,0])
        th=np.transpose(th,[1,2,0])
        te=np.transpose(te,[1,2,3,4,0])
        te_mask=np.transpose(te_mask,[1,2,3,4,0])

    return delta,tx,ty,tw,th,te,te_mask

def draw_bbx(img,img_pc,rx,ry,rw,rh,threshold=0.7):
    color=(0,255,0)
    valid_idxs=np.where(img_pc>=threshold,1,0)
    ks,iys,ixs=np.nonzero(valid_idxs)
    h,w,_=img.shape
    thickness=int(min(h,w)/100)
    for k,iy,ix in zip(ks,iys,ixs):
        x=rx[k][iy][ix]
        y=ry[k][iy][ix]
        w=rw[k][iy][ix]
        h=rh[k][iy][ix]
        img=cv2.rectangle(img,(int(x-w//2),int(y-h//2)),(int(x+w//2),int(y+h//2)),color,thickness=thickness)
    return img

def draw_edge(img,img_e,rx,ry,rw,rh,hnei,wnei,hout,wout,limbs,threshold=0.7):
    color=(255,0,0)
    valid_idxs=np.where(img_e>=threshold,1,0)
    ls,niys,nixs,iys,ixs=np.nonzero(valid_idxs)
    h,w,_=img.shape
    thickness=int(min(h,w)/100)
    for l,niy,nix,iy,ix in zip(ls,niys,nixs,iys,ixs):
        s_id=limbs[l][0]
        d_id=limbs[l][1]
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
        img=cv2.line(img,(int(src_x),int(src_y)),(int(dst_x),int(dst_y)),color,thickness=thickness)
    return img

def draw_results(img,predicts,targets,parts,limbs,save_dir,threshold=0.3,name="",is_train=True,data_format="channels_first"):
    pc,px,py,pw,ph,pe=predicts
    if(data_format=="channels_last"):
        pc=np.transpose(pc,[0,3,1,2])
        px=np.transpose(px,[0,3,1,2])
        py=np.transpose(py,[0,3,1,2])
        pw=np.transpose(pw,[0,3,1,2])
        ph=np.transpose(ph,[0,3,1,2])
        pe=np.transpose(pe,[0,5,1,2,3,4])
    else:
        img=np.transpose(img,[0,2,3,1])
    if(is_train):
        tc,tx,ty,tw,th,te,_=targets
        if(data_format=="channels_last"):
            tc=np.transpose(tc,[0,3,1,2])
            tx=np.transpose(tx,[0,3,1,2])
            ty=np.transpose(ty,[0,3,1,2])
            tw=np.transpose(tw,[0,3,1,2])
            th=np.transpose(th,[0,3,1,2])
            te=np.transpose(te,[0,5,1,2,3,4])

    img=np.clip(img*255.0,0,255).astype(np.uint8)
    batch_size,hin,win,_=img.shape
    batch_size,K,hout,wout=px.shape
    batch_size,L,hnei,wnei,hout,wout=pe.shape
    if(is_train):
        rtx,rty,rtw,rth=restore_coor(tx,ty,tw,th,win,hin,wout,hout)
        rpx,rpy,rpw,rph=restore_coor(px,py,pw,ph,win,hin,wout,hout)
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
            img_pd=draw_edge(img_pd,pe[b_idx],rpx[b_idx],rpy[b_idx],rpw[b_idx],rph[b_idx],hnei,wnei,hout,wout,limbs,threshold)
            sub_plot.imshow(img_pd)
            #draw ground truth image
            sub_plot=fig.add_subplot(2,2,3)
            sub_plot.set_title("True image")
            img_gt=img[b_idx].copy()
            img_gt=draw_bbx(img_gt,tc[b_idx],rtx[b_idx],rty[b_idx],rtw[b_idx],rth[b_idx],threshold)
            img_gt=draw_edge(img_gt,te[b_idx],rtx[b_idx],rty[b_idx],rtw[b_idx],rth[b_idx],hnei,wnei,hout,wout,limbs,threshold)
            sub_plot.imshow(img_gt)
            #save figure
            plt.savefig(os.path.join(save_dir, '%s_%d.png' % (name, b_idx)), dpi=300)
            plt.close()
    else:
        rpx,rpy,rpw,rph=px,py,pw,ph
        for b_idx in range(0,batch_size):
            fig=plt.figure(figsize=(8,8))
            #draw original image
            sub_plot=fig.add_subplot(1,2,1)
            sub_plot.set_title("Originl image")
            sub_plot.imshow(img[b_idx].copy())
            #draw predict image
            sub_plot=fig.add_subplot(1,2,2)
            sub_plot.set_title("Predict image")
            img_pd=img[b_idx].copy()
            img_pd=draw_bbx(img_pd,pc[b_idx],rpx[b_idx],rpy[b_idx],rpw[b_idx],rph[b_idx],threshold)
            img_pd=draw_edge(img_pd,pe[b_idx],rpx[b_idx],rpy[b_idx],rpw[b_idx],rph[b_idx],hnei,wnei,hout,wout,limbs,threshold)
            sub_plot.imshow(img_pd)
            #save figure
            plt.savefig(os.path.join(save_dir, '%s_%d.png' % (name, b_idx)), dpi=300)
            plt.close()

def restore_coor(x,y,w,h,win,hin,wout,hout,data_format="channels_first"):
        grid_size_x=win/wout
        grid_size_y=hin/hout
        grid_x,grid_y=tf.meshgrid(np.arange(wout).astype(np.float32),np.arange(hout).astype(np.float32))
        if(data_format=="channels_last"):
            grid_x=grid_x[...,np.newaxis]
            grid_y=grid_y[...,np.newaxis]
        rx=(x+grid_x)*grid_size_x
        ry=(y+grid_y)*grid_size_y
        rw=w*win
        rh=h*hin
        return rx,ry,rw,rh

def cal_iou(bbx1,bbx2):
    x1,y1,w1,h1=bbx1
    x2,y2,w2,h2=bbx2
    area1=w1*h1
    area2=w2*h2
    inter_x=tf.nn.relu(tf.minimum(x1+w1/2,x2+w2/2)-tf.maximum(x1-w1/2,x2-w2/2))
    inter_y=tf.nn.relu(tf.minimum(y1+h1/2,y2+h2/2)-tf.maximum(y1-h1/2,y2-h2/2))
    inter_area=inter_x*inter_y
    union_area=area1+area2-inter_area
    return inter_area/union_area

def non_maximium_supress(bbxs,scores,thres):
    # bbxs=[4,bipartnum]
    bbxs_num=bbxs.shape[0]
    idx=np.linspace(start=0,stop=bbxs_num-1,num=bbxs_num).astype(np.int)[:,np.newaxis]
    idxed_bbxs=np.concatenate([bbxs,idx],axis=1)
    chosen_idxs=[]
    left_bbxs=idxed_bbxs
    left_scores=scores
    for _ in range(0,bbxs_num):
        #sort most convinced bbx
        sort_idx=np.argsort(-left_scores,axis=0)
        left_scores=left_scores[sort_idx]
        left_bbxs=left_bbxs[sort_idx,:]
        maxconf_bbx=left_bbxs[0]
        chosen_idxs.append(maxconf_bbx[4])
        #calculate iou with other bbxs
        #ious is the size [left_bbxnum]
        ious=cal_iou(maxconf_bbx[0:4],left_bbxs[:,0:4].transpose())
        left_idx=np.where(ious<thres)[0]
        if(len(left_idx)==0):
            break
        else:
            left_scores=left_scores[left_idx]
            left_bbxs=left_bbxs[left_idx,:]
    chosen_idxs=np.array(chosen_idxs).astype(np.int)
    #print(f"test chosen_idxs")
    return chosen_idxs

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

from .processor import PostProcessor
from ..common import tf_repeat,TRAIN,MODEL,DATA


def preprocess(annos,bbxs,model_hin,modeL_win,model_hout,model_wout,model_hnei,model_wnei,parts,limbs,data_format="channels_first"):
    '''preprocess function of poseproposal class models

    take keypoints annotations, bounding boxs annotatiosn, model input height and width, model limbs neighbor area height,
    model limbs neighbor area width and dataset type
    return the constructed targets of delta,tx,ty,tw,th,te,te_mask

    Parameters
    ----------
    arg1 : list
        a list of keypoint annotations, each annotation is a list of keypoints that belongs to a person, each keypoint follows the
        format (x,y), and x<0 or y<0 if the keypoint is not visible or not annotated.
        the annotations must from a known dataset_type, other wise the keypoint and limbs order will not be correct.

    arg2 : list
        a list of bounding box annotations, each bounding box is of format [x,y,w,h]

    arg3 : Int
        height of the model input

    arg4 : Int
        width of the model input

    arg5 : Int
        height of the model output

    arg6 : Int
        width of the model output

    arg7 : Int
        model limbs neighbor area height, determine the neighbor area to macth limbs,
        see pose propsal paper for detail information

    arg8 : Int
        model limbs neighbor area width, determine the neighbor area to macth limbs,
        see pose propsal paper for detail information

    arg9 : Config.DATA
        a enum value of enum class Config.DATA
        dataset_type where the input annotation list from, because to generate correct
        conf_map and paf_map, the order of keypoints and limbs should be awared.

    arg10 : string
        data format speficied for channel order
        available input:
        'channels_first': data_shape C*H*W
        'channels_last': data_shape H*W*C

    Returns
    -------
    list
        including 7 elements
        delta: keypoint confidence feature map, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        tx: keypoints bbx center x coordinates, divided by gridsize, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        ty: keypoints bbx center y coordinates, divided by gridsize, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        tw: keypoints bbxs width w, divided by image width, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        th: keypoints bbxs height h, divided by image width, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        te: edge confidence feature map,  shape [C,H,W,Hnei,Wnei](channels_first) or [H,W,Hnei,Wnei,C](channels_last)
        te_mask: mask of edge confidence feature map, used for loss caculation,
        shape [C,H,W,Hnei,Wnei](channels_first) or [H,W,Hnei,Wnei,C](channels_last)
    '''
    delta,tx,ty,tw,th,te,te_mask=get_pose_proposals(annos,bbxs,model_hin,modeL_win,model_hout,\
        model_wout,model_hnei,model_wnei,parts,limbs,img_mask=None,data_format=data_format)
    return delta,tx,ty,tw,th,te,te_mask

def postprocess(predicts,parts,limbs,data_format="channels_first",colors=None):
    '''postprocess function of poseproposal class models

    take model predicted feature maps of delta,tx,ty,tw,th,te,te_mask,
    output parsed human objects, each one contains all detected keypoints of the person

    Parameters
    ----------
    arg1 : list
        a list of model output: delta,tx,ty,tw,th,te,te_mask
        delta: keypoint confidence feature map, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        tx: keypoints bbx center x coordinates, divided by gridsize, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        ty: keypoints bbx center y coordinates, divided by gridsize, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        tw: keypoints bbxs width w, divided by image width, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        th: keypoints bbxs height h, divided by image width, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        te: edge confidence feature map,  shape [C,H,W,Hnei,Wnei](channels_first) or [H,W,Hnei,Wnei,C](channels_last)
        te_mask: mask of edge confidence feature map, used for loss caculation,
        shape [C,H,W,Hnei,Wnei](channels_first) or [H,W,Hnei,Wnei,C](channels_last)

    arg2: Config.DATA
        a enum value of enum class Config.DATA
        dataset_type where the input annotation list from, because to generate correct
        conf_map and paf_map, the order of keypoints and limbs should be awared.

    arg3 : string
        data format speficied for channel order
        available input:
        'channels_first': data_shape C*H*W
        'channels_last': data_shape H*W*C

    Returns
    -------
    list
        contain object of humans,see Model.Human for detail information of Human object
    '''
    pc,pi,px,py,pw,ph,pe=predicts
    for x in [pc,pc,px,py,pw,ph,pe]:
        if(type(x)!=np.ndarray):
            x=x.numpy()
    if(colors==None):
        colors=[[255,0,0]]*len(parts)
    post_processor=PostProcessor(parts,limbs,colors)
    if(data_format=="channels_last"):
        pc=np.transpose(pc,[2,0,1])
        pi=np.transpose(pi,[2,0,1])
        px=np.transpose(px,[2,0,1])
        py=np.transpose(py,[2,0,1])
        pw=np.transpose(pw,[2,0,1])
        ph=np.transpose(ph,[2,0,1])
        pe=np.transpose(pe,[4,0,1,2,3])
    humans=post_processor.process(pc,pi,px,py,pw,ph,pe)
    return humans

def visualize(img,predicts,parts,limbs,save_name="bbxs",save_dir="./save_dir/vis_dir",data_format="channels_first",save_tofile=True):
    '''visualize function of poseproposal class models

    take model predicted feature maps of delta,tx,ty,tw,th,te,te_mask, output visualized image.
    the image will be saved at 'save_dir'/'save_name'_visualize.png

    Parameters
    ----------
    arg1 : numpy array
        image

    arg2 : list
        a list of model output: delta,tx,ty,tw,th,te,te_mask
        delta: keypoint confidence feature map, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        tx: keypoints bbx center x coordinates, divided by gridsize, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        ty: keypoints bbx center y coordinates, divided by gridsize, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        tw: keypoints bbxs width w, divided by image width, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        th: keypoints bbxs height h, divided by image width, shape [C,H,W](channels_first) or [H,W,C](channels_last)
        te: edge confidence feature map,  shape [C,H,W,Hnei,Wnei](channels_first) or [H,W,Hnei,Wnei,C](channels_last)
        te_mask: mask of edge confidence feature map, used for loss caculation,
        shape [C,H,W,Hnei,Wnei](channels_first) or [H,W,Hnei,Wnei,C](channels_last)

    arg3: Config.DATA
        a enum value of enum class Config.DATA
        dataset_type where the input annotation list from

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
    pc,pi,px,py,pw,ph,pe=predicts
    for x in [pc,pc,px,py,pw,ph,pe]:
        if(type(x)!=np.ndarray):
            x=x.numpy()
    if(data_format=="channels_last"):
        pc=np.transpose(pc,[2,0,1])
        pi=np.transpose(pi,[2,0,1])
        px=np.transpose(px,[2,0,1])
        py=np.transpose(py,[2,0,1])
        pw=np.transpose(pw,[2,0,1])
        ph=np.transpose(ph,[2,0,1])
        pe=np.transpose(pe,[4,0,1,2,3])
    elif(data_format=="channels_first"):
        img=np.transpose(img,[1,2,0])
    _,model_hnei,model_wnei,model_hout,model_wout=pe.shape
    os.makedirs(save_dir,exist_ok=True)
    ori_img=np.clip(img*255.0,0.0,255.0).astype(np.uint8)
    #show input image
    fig=plt.figure(figsize=(8,8))
    a=fig.add_subplot(2,2,1)
    a.set_title("input image")
    plt.imshow(ori_img)
    
    vis_img=ori_img.copy()
    #show parts
    vis_parts_img=draw_bbx(vis_img,pc,px,py,pw,ph,threshold=0.7)
    a=fig.add_subplot(2,2,2)
    a.set_title("visualized kpt result")
    plt.imshow(vis_parts_img)
    #show edges
    vis_limbs_img=draw_edge(vis_img,pe,px,py,pw,ph,model_hnei,model_wnei,model_hout,model_wout,limbs,threshold=0.7)
    a=fig.add_subplot(2,2,3)
    a.set_title("visualized limb result")
    plt.imshow(vis_limbs_img)
    #save result
    if(save_tofile):
        plt.savefig(f"{save_dir}/{save_name}_visualize.png")
        plt.close()
    return vis_parts_img,vis_limbs_img