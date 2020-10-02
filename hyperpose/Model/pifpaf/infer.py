
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
    def __init__(self,parts,limbs,colors,debug=False,stride=8,thresh_pif=0.1,thresh_paf=0.1,thresh_ref_pif=0.1,thresh_ref_paf=0.1,\
        reduction=2,min_scale=4,greedy_match=True,reverse_match=True):
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
        self.by_source=defaultdict()
        self.greedy_match=greedy_match
        self.reverse_match=reverse_match

        for limb_idx,(src_idx,dst_idx) in enumerate(self.limbs):
            self.by_source[src_idx][dst_idx]=(limb_idx,True)
            self.by_source[dst_idx][src_idx]=(limb_idx,False)
    
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
    def check_occupy(self,occupied,pos_idx,x,y):
        _,field_h,field_w=occupied.shape
        x,y=x/self.reduction,y/self.reduction
        if(x<0 or x>=field_w or y>0 or y>=field_h):
            return True
        if(occupied[pos_idx,y,x]!=0):
            return True
        else:
            return False
    
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
            return None
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
        find_connection=self.find_connection(forward_cons,x,y,scale,connection_method)
        
            



    
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
                match_xycs=self.get_connection_value(ann,src_idx,dst_idx,forward_slist,backward_list,reverse_match=reverse_match)

        #initially add joints to frontier
        for pos_idx in range(0,self.n_pos):
            if(ann[pos_idx,0]>0):
                add_frontier(ann,pos_idx)
        #recurrently finding the match connections
        while True:
            find_match=get_frontier(ann)




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
        for c,x,y,scale,pos_idx in seeds:
            if(self.check_occupy(occupied,pos_idx,x,y)):
                continue
            ann=np.zeros(shape=(self.n_pos,4))
            ann[pos_idx]=c,x,y,scale
            self.grow(ann,forward_list,backward_list,reverse_match=self.reverse_match)

