
import os
import cv2
import json
import heapq
import numpy as np
import tensorflow as tf
import scipy.stats as st
from .utils import get_hr_conf
from ..human import Human,BodyPart
from collections import defaultdict

class Post_Processor:
    def __init__(self,parts,limbs,colors,stride=8,thresh_pif=0.1,thresh_paf=0.1,thresh_ref_pif=0.1,thresh_ref_paf=0.1,\
        reduction=2,min_scale=4,greedy_match=True,reverse_match=True,debug=False):
        self.parts=parts
        self.limbs=limbs
        self.colors=colors
        self.n_pos=len(self.parts)
        self.n_limbs=len(self.limbs)
        self.stride=stride
        self.thresh_pif=thresh_pif
        self.thresh_paf=thresh_paf
        self.thresh_ref_pif=thresh_ref_pif
        self.thresh_ref_paf=thresh_ref_paf
        self.reduction=reduction
        self.min_scale=min_scale
        self.greedy_match=greedy_match
        self.reverse_match=reverse_match
        self.by_source=defaultdict()
        for limb_idx,(src_idx,dst_idx) in enumerate(self.limbs):
            self.by_source[src_idx][dst_idx]=(limb_idx,True)
            self.by_source[dst_idx][src_idx]=(limb_idx,False)
        self.part_num_thresh=4
        self.score_thresh=0.1
        self.debug=debug
        #TODO:whether add score weight for each parts
    
    
    #convert vector field to scalar
    def field_to_scalar(self,vec_x,vec_y,scalar_map):
        #scalar_map shape:[height,width]
        #vec_map shape:[2,vec_num]
        h,w=scalar_map.shape
        vec_num=vec_x.shape[1]
        ret_scalar=np.zeros(vec_num)
        for vec_idx in range(0,vec_num):
            x,y=np.round(vec_x[vec_idx]),np.round(vec_y[vec_idx])
            if(x>=0 and x<w and y>=0 and y<h):
                ret_scalar[vec_idx]=scalar_map[x,y]
        return ret_scalar
    
    #check whether the position is occupied
    def check_occupy(self,occupied,pos_idx,x,y,reduction=2):
        _,field_h,field_w=occupied.shape
        x,y=np.round(x/reduction),np.round(y/reduction)
        if(x<0 or x>=field_w or y>0 or y>=field_h):
            return True
        if(occupied[pos_idx,y,x]!=0):
            return True
        else:
            return False
    
    #mark the postion as occupied
    def put_occupy(self,occupied,pos_idx,x,y,scale,reduction=2,min_scale=4,value=1):
        _,field_h,field_w=occupied.shape
        x,y=np.round(x/reduction),np.round(y/reduction)
        size=np.round(max(min_scale/reduction,scale/reduction))
        min_x=max(0,int(x-size))
        max_x=max(min_x+1,min(field_w,int(x+size)+1))
        min_y=max(0,int(y-size))
        max_y=max(min_y+1,min(field_h,int(y+size)+1))
        occupied[pos_idx,min_y:max_y,min_x:max_x]+=value
        return occupied
    
    #keypoint-wise nms
    def kpt_nms(self,annotations):
        max_x=int(max([np.max(ann[:,1]) for ann in annotations])+1)
        max_y=int(max([np.max(ann[:,2]) for ann in annotations])+1)
        occupied=np.zeros(shape=(self.n_pos,max_y,max_x))
        annotations=sorted(annotations,key=lambda ann: -np.sum(ann[:,0]))
        for ann in annotations:
            for pos_idx in range(0,self.n_pos):
                _,x,y,scale=ann[pos_idx]
                if(self.check_occupy(occupied,pos_idx,x,y,reduction=2)):
                    ann[pos_idx,0]=0
                else:
                    self.put_occupy(occupied,pos_idx,x,y,scale,reduction=2,min_scale=4)
        annotations=sorted(annotations,key=lambda ann: -np.sum(ann[:,0]))
        return annotations
    
    #get closest matching connection and blend them
    def find_connection(self,connections,x,y,scale,connection_method="blend",thresh_second=0.01):
        sigma_filter=2.0*scale
        sigma_gaussian=0.25*(scale**2)
        first_idx,first_score=-1,0.0
        second_idx,second_score=-1,0.0
        #traverse connections to find the highest score connection weighted by distance
        score_f,src_x,src_y,src_scale,dst_x,dst_y,dst_scale=connections
        con_num=score_f.shape[1]
        for con_idx in range(0,con_num):
            con_score=score_f[con_idx]
            con_src_x,con_src_y,_=src_x[con_idx],src_y[con_idx],src_scale[con_idx]
            #ignore connections with src_kpts too distant 
            if(x<con_src_x-sigma_filter or x>con_src_x+sigma_filter):
                continue
            if(y<con_src_y-sigma_filter or y>con_src_y+sigma_filter):
                continue
            distance=(con_src_x-x)**2+(con_src_y-y)**2
            w_score=np.exp(-0.5*distance/sigma_gaussian)*con_score
            #replace to find the first and second match connections
            if(w_score>first_score):
                second_idx=first_idx
                second_score=first_score
                first_idx=con_idx
                first_score=w_score
            elif(w_score>second_score):
                second_idx=con_idx
                second_score=w_score
        #not find match connections
        if(first_idx==-1 or first_score==0.0):
            return 0.0,0.0,0.0,0.0
        #method max:
        if(connection_method=="max"):
            return first_score,dst_x[first_idx],dst_y[first_idx],dst_scale[first_idx]
        #method blend:
        elif(connection_method=="blend"):
            #ignore second connection with score too slow
            if(second_idx==-1 or second_score<thresh_second or second_score<0.5*first_score):
                return first_score*0.5,dst_x[first_idx],dst_y[first_idx],dst_scale[first_idx]
            #ignore second connection too distant from the first one
            dist_first_second=(dst_x[first_idx]-dst_x[second_idx])**2+(dst_y[first_idx]-dst_y[second_idx])**2
            if(dist_first_second>(dst_scale[first_idx]**2/4.0)):
                return first_score*0.5,dst_x[first_idx],dst_y[first_idx],dst_scale[first_idx]
            #otherwise return the blended two connection
            blend_score=0.5*(first_score+second_score)
            blend_x=(dst_x[first_idx]*first_score+dst_x[second_idx]*second_score)/(first_score+second_score)
            blend_y=(dst_y[first_idx]*first_score+dst_y[second_idx]*second_score)/(first_score+second_score)
            blend_scale=(dst_scale[first_idx]*first_score+dst_scale[second_idx]*second_score)/(first_score+second_score)
            return blend_score,blend_x,blend_y,blend_scale
    
    #get connection given a part, forwad_list and backward_list generated from paf maps 
    def get_connection(self,ann,src_idx,dst_idx,forward_list,backward_list,connection_method="blend",reverse_match=True):
        limb_idx,forward_flag=self.by_source[src_idx][dst_idx]
        if(forward_flag):
            forward_cons,backward_cons=forward_list[limb_idx],backward_list[limb_idx]
        else:
            forward_cons,backward_cons=backward_list[limb_idx],forward_list[limb_idx]
        c,x,y,scale=ann[src_idx]
        #forward matching
        fc,fx,fy,fscale=self.find_connection(forward_cons,x,y,scale,connection_method=connection_method)
        if(fc==0.0):
            return 0.0,0.0,0.0,0.0
        merge_score=np.sqrt(fc*c)
        #reverse matching
        if(reverse_match):
            rc,rx,ry,_=self.find_connection(backward_cons,fx,fy,fscale,connection_method=connection_method)
            #couldn't find a reverse one
            if(rc==0.0):
                return 0.0,0.0,0.0,0.0
            #reverse finding is distant from the orginal founded one
            if abs(x-rx)+abs(y-ry)>scale:
                return 0.0,0.0,0.0,0.0
        #successfully found connection
        return merge_score,fx,fy,fscale
    
    #greedy matching pif seeds with forward and backward connections generated from paf maps
    def grow(self,ann,forward_list,backward_list,reverse_match=True):
        frontier = []
        in_frontier = set()
        #add the point to assemble frontier
        def add_frontier(ann,src_idx):
            #traverse all the part that the current part connect to
            for dst_idx,(_,_) in self.by_source[src_idx].items():
                #ignore points that already assigned
                if(ann[dst_idx,0]>0):
                    continue
                #ignore limbs that already in the frontier
                if((src_idx,dst_idx) in in_frontier):
                    continue
                #otherwise put it into frontier
                max_possible_score=np.sqrt(ann[src_idx,0])
                heapq.heappush(frontier,(-max_possible_score,src_idx,dst_idx))
                in_frontier.add((src_idx,dst_idx))
        
        #find matching connections from frontier
        def get_frontier(ann):
            while frontier:
                pop_frontier=heapq.heappop(frontier)
                _,src_idx,dst_idx=pop_frontier
                #ignore points that assigned by other frontier
                if(ann[dst_idx,0]>0.0):
                    continue
                #find conection
                fc,fx,fy,fscale=self.get_connection_value(ann,src_idx,dst_idx,forward_list,backward_list,reverse_match=reverse_match)
                if(fc==0.0):
                    continue
                return fc,fx,fy,fscale,src_idx,dst_idx
            return None

        #initially add joints to frontier
        for pos_idx in range(0,self.n_pos):
            if(ann[pos_idx,0]>0.0):
                add_frontier(ann,pos_idx)
        #recurrently finding the match connections
        while True:
            find_match=get_frontier(ann)
            if(find_match==None):
                break
            score,x,y,scale,_,dst_idx=find_match
            if(ann[dst_idx,0]>0.0):
                continue
            ann[dst_idx,0]=score
            ann[dst_idx,1]=x
            ann[dst_idx,2]=y
            ann[dst_idx,3]=scale
            add_frontier(ann,dst_idx)
        #finished matching a person
        return ann

    def process(self,pif_maps,paf_maps):
        #shape:
        #conf_map:[field_num,hout,wout]
        #vec_map:[field_num,2,hout,wout]
        #scale_map:[field_num,hout,wout]
        stride=self.stride
        #decode pif_maps,paf_maps
        pif_conf,pif_vec,_,pif_scale=pif_maps
        paf_conf,paf_src_vec,paf_dst_vec,_,_,paf_src_scale,paf_dst_scale=paf_maps
        #get pif_hr_conf
        pif_hr_conf=get_hr_conf(pif_conf,pif_vec,pif_scale,stride=self.stride,thresh=self.thresh_pif)
        #generate pose seeds according to refined pif_conf
        seeds=[]
        for pos_idx in range(0,self.n_pos):
            seeds=[]
            mask_conf=pif_conf[pos_idx]>self.thresh_pif
            c=pif_conf[pos_idx,mask_conf]
            x=pif_vec[pos_idx,0,mask_conf]*stride
            y=pif_vec[pos_idx,1,mask_conf]*stride
            scale=pif_scale[pos_idx,mask_conf]
            hr_c=self.field_to_scalar(x,y,pif_hr_conf[pos_idx])
            ref_c=0.9*hr_c+0.1*c
            mask_ref_conf=ref_c>self.thresh_ref_pif
            seeds.append((ref_c[mask_ref_conf],x[mask_ref_conf],y[mask_ref_conf],scale[mask_ref_conf],pos_idx))
        sorted(seeds,reverse=True)
        #generate connection seeds according to paf_map
        cif_floor=0.1
        forward_list=[]
        backward_list=[]
        for limb_idx in range(0,self.n_limbs):
            src_idx,dst_idx=self.limbs[limb_idx]
            mask_conf=paf_conf[limb_idx]>self.thresh_paf
            score=paf_conf[limb_idx,mask_conf]
            src_x=paf_src_vec[limb_idx,0,mask_conf]*stride
            src_y=paf_src_vec[limb_idx,1,mask_conf]*stride
            dst_x=paf_dst_vec[limb_idx,0,mask_conf]*stride
            dst_y=paf_dst_vec[limb_idx,1,mask_conf]*stride
            src_scale=paf_src_scale[limb_idx,mask_conf]*stride
            dst_scale=paf_dst_scale[limb_idx,mask_conf]*stride
            #generate backward (merge score with the src pif_score)
            cifhr_b=self.field_to_scalar(src_x,src_y,pif_hr_conf[src_idx])
            score_b=score*(cif_floor+(1-cif_floor)*cifhr_b)
            mask_b=score_b>self.thresh_ref_paf
            backward_list.append([score_b[mask_b],dst_x[mask_b],dst_y[mask_b],dst_scale[mask_b],src_x[mask_b],src_y[mask_b],src_scale[mask_b]])
            #generate forward connections (merge score with the dst pif_score)
            cifhr_f=self.field_to_scalar(dst_x,dst_y,pif_hr_conf[dst_idx])
            score_f=score*(cif_floor+(1-cif_floor)*cifhr_f)
            mask_f=score_f>self.thresh_ref_paf
            forward_list.append([score_f[mask_f],src_x[mask_f],src_y[mask_f],src_scale[mask_f],dst_x[mask_f],dst_y[mask_f],dst_scale[mask_f]])
        #greedy assemble
        occupied=np.zeros(shape=(self.n_pos,pif_hr_conf.shape[1]/self.reduction,pif_hr_conf.shape[2]/self.reduction))
        annotations=[]
        for c,x,y,scale,pos_idx in seeds:
            if(self.check_occupy(occupied,pos_idx,x,y,reduction=self.reduction)):
                continue
            #ann meaning: ann[0]=conf ann[1]=x ann[2]=y ann[3]=scale
            ann=np.zeros(shape=(self.n_pos,4))
            ann[pos_idx]=c,x,y,scale
            ann=self.grow(ann,forward_list,backward_list,reverse_match=self.reverse_match)
            annotations.append(ann)
            #put the ann into occupacy
            for ann_pos_idx in range(0,self.n_pos):
                occupied=self.put_occupy(occupied,ann_pos_idx,ann[ann_pos_idx,1],ann[ann_pos_idx,2],ann[ann_pos_idx,3],\
                    reduction=self.reduction,min_scale=self.min_scale)
        #point-wise nms
        annotations=self.kpt_nms(annotations)
        #convert to humans
        ret_humans=[]
        for ann_idx,ann in enumerate(annotations):
            ret_human=Human(parts=self.parts,limbs=self.limbs,colors=self.colors)
            for pos_idx in range(0,self.n_pos):
                score,x,y,scale=ann[pos_idx]
                ret_human.body_parts[pos_idx]=BodyPart(parts=self.parts,u_idx=f"{ann_idx}-{pos_idx}",part_idx=pos_idx,\
                    x=x,y=y,score=score)
            #check for num
            if(ret_human.get_partnum()<self.part_num_thresh):
                continue
            if(ret_human.get_score()<self.score_thresh):
                continue
            ret_humans.append(ret_human)
        if(self.debug):
            print(f"total {len(ret_humans)} human detected!")
        return ret_humans





