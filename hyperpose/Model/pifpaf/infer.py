
import os
import cv2
import json
import numpy as np
import tensorflow as tf
import scipy.stats as st
from .utils import get_hr_conf
from ..human import Human,BodyPart
from collections import defaultdict

class Post_Processor:
    def __init__(self,parts,limbs,colors,debug=False):
        self.parts=parts
        self.limbs=limbs
        self.colors=colors
        self.n_pos=len(self.parts)
        self.n_limbs=len(self.limbs)
        self.by_source=defaultdict()
        for limb_idx,(src_idx,dst_idx) in enumerate(self.limbs):
            self.by_source[src_idx][dst_idx]=(limb_idx,True)
            self.by_source[dst_idx][src_idx]=(limb_idx,False)
        
    def field_to_scalar(self,vec_map,scalar_map):
        #scalar_map shape:[height,width]
        #vec_map shape:[2,vec_num]
        vec_num=vec_map.shape[1]
        for vec_idx in range(0,vec_num):



    def process(self,pif_maps,paf_maps,stride=8,thresh_pif=0.1,thresh_paf=0.1):
        #shape:
        #conf_map:[field_num,1,hout,wout]
        #vec_map:[field_num,2,hout,wout]
        #scale_map:[field_num,1,hout,wout]
        #decode pif_maps,paf_maps
        pif_conf,pif_vec,_,pif_scale=pif_maps
        paf_conf,paf_src_vec,paf_dst_vecm_,_,paf_src_scale,paf_dst_scale=paf_maps
        #get pif_hr_conf
        pif_hr_conf=get_hr_conf(pif_conf,pif_vec,pif_scale,stride=stride,thresh=thresh_pif)
        #refine pif_conf according to pif_hr_conf
        for pos_idx in range(0,self.n_pos):
            conf_mask=pif_conf[pos_idx]>thresh_pif
            c=pif_conf[pos_idx,conf_mask]
            x,y=pif_vec[pos_idx,conf_mask]*stride
            s=pif_scale[pos_idx,conf_mask]
            


        #get pif seeds for pose generation
