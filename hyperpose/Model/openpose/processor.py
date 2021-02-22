import os
import cv2
import json
import numpy as np
import tensorflow as tf
import scipy.stats as st
from .utils import get_heatmap,get_vectormap
from ..human import Human,BodyPart

class PreProcessor:
    def __init__(self,parts,limbs,hin,win,hout,wout,colors=None,data_format="channels_first"):
        self.hin=hin
        self.win=win
        self.hout=hout
        self.wout=wout
        self.parts=parts
        self.limbs=limbs
        self.data_format=data_format
        self.colors=colors if (colors!=None) else (len(self.parts)*[[0,255,0]])

    def process(self,annos,mask_valid):
        heatmap = get_heatmap(annos, self.hin, self.win, self.hout, self.wout, self.parts, self.limbs, data_format=self.data_format)
        vectormap = get_vectormap(annos, self.hin, self.win, self.hout, self.wout, self.parts, self.limbs, data_format=self.data_format)
        return heatmap,vectormap


class PostProcessor:
    def __init__(self,parts,limbs,hin,win,hout,wout,colors=None,thresh_conf=0.05,thresh_vec=0.05,thresh_vec_cnt=6,\
                    step_paf=10,thresh_criterion2=0,thresh_part_cnt=4,thresh_human_score=0.3,data_format="channels_first",debug=False):
        self.cur_id=0
        self.parts=parts
        self.limbs=limbs
        self.colors=colors if (colors!=None) else (len(self.parts)*[[0,255,0]])
        self.n_pos=len(self.parts)
        self.n_limb=len(self.limbs)
        self.thresh_conf=thresh_conf
        self.thresh_vec=thresh_vec
        self.thresh_vec_cnt=thresh_vec_cnt
        self.step_paf=step_paf
        self.thresh_criterion2=thresh_criterion2
        self.thresh_part_cnt=thresh_part_cnt
        self.thresh_human_score=thresh_human_score
        self.data_format=data_format
        self.debug=debug
    
    def process(self,conf_map,paf_map,img_h,img_w,data_format="channels_first"):
        if(data_format=="channels_first"):
            conf_map=np.transpose(conf_map,[1,2,0])
            paf_map=np.transpose(paf_map,[1,2,0])
        #conf_map
        if(conf_map.shape[0]!=img_h or conf_map.shape[1]!=img_w):
            conf_map=cv2.resize(conf_map,(img_w,img_h))
        conf_map=conf_map[np.newaxis,:,:,:]
        #paf_map
        if(paf_map.shape[0]!=img_h or paf_map.shape[1]!=img_w):
            paf_map=cv2.resize(paf_map,(img_w,img_h))
        paf_map=paf_map[np.newaxis,:,:,:]
        peak_map=self.get_peak_map(conf_map)
        humans=self.process_paf(peak_map[0],conf_map[0],paf_map[0])
        return humans
    
    def get_peak_map(self,conf_map):
        def _gauss_smooth(origin):
            sigma=3.0
            kernel_size=5
            smoothed=np.zeros(shape=origin.shape)
            channel_num=origin.shape[-1]
            for channel_idx in range(0,channel_num):
                smoothed[0,:,:,channel_idx]=cv2.GaussianBlur(origin[0,:,:,channel_idx],\
                    ksize=(kernel_size,kernel_size),sigmaX=sigma,sigmaY=sigma)
            return smoothed

        smoothed = _gauss_smooth(conf_map)
        max_pooled = tf.nn.pool(smoothed, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
        return tf.where(tf.equal(smoothed, max_pooled), conf_map, tf.zeros_like(conf_map)).numpy()
    
    def process_paf(self,peak_map,conf_map,paf_map):
        #filter valid peaks
        peaks=[[] for part_idx in range(0,self.n_pos)]
        all_peaks=[]
        peak_ys,peak_xs,part_idxs=np.where(peak_map>self.thresh_conf)
        for peak_idx,(part_idx,peak_y,peak_x) in enumerate(zip(part_idxs,peak_ys,peak_xs)):
            peak_score=conf_map[peak_y,peak_x,part_idx]
            peaks[part_idx].append(Peak(peak_idx,part_idx,peak_y,peak_x,peak_score))
            all_peaks.append(Peak(peak_idx,part_idx,peak_y,peak_x,peak_score))
        #start candidate connection
        if(self.debug):
            print("peak debug:")
            for part_idx in range(0,self.n_pos):
                print(f"found peak_{part_idx}:{len(peaks[part_idx])}")
            for part_idx in range(0,self.n_pos):
                print(f"peak {self.parts(part_idx)}:")
                for peak in peaks[part_idx]:
                    print(f"peak {self.parts(part_idx)} id:{peak.idx} x:{peak.x} y:{peak.y} score:{peak.score}")
                print()

        candidate_limbs=[[] for limb_idx in range(0,len(self.limbs))]
        for limb_idx,limb in enumerate(self.limbs):
            src_idx,dst_idx=limb
            peak_src_list=peaks[src_idx]
            peak_dst_list=peaks[dst_idx]
            if((len(peak_src_list)==0) or(len(peak_dst_list)==0)):
                continue
            self.debug_print(f"candidating: src:{self.parts(src_idx)} dst:{self.parts(dst_idx)}")
            for peak_src in peak_src_list:
                for peak_dst in peak_dst_list:
                    #calculate paf vector
                    vec_src=np.array([peak_src.y,peak_src.x])
                    vec_dst=np.array([peak_dst.y,peak_dst.x])
                    vec_limb=vec_dst-vec_src
                    lenth=np.sqrt(np.sum(vec_limb**2))
                    if(lenth<1e-12):
                        continue
                    vec_limb=vec_limb/lenth
                    paf_vectors=self.get_paf_vectors(limb_idx,vec_src,vec_dst,paf_map)
                    #calculate criterion
                    criterion1=0
                    scores=0.0
                    for step in range(self.step_paf):
                        score=np.sum(vec_limb*paf_vectors[step])
                        if(score>=self.thresh_vec):
                            criterion1+=1
                        scores+=score
                    criterion2=scores/self.step_paf+min(0.0,0.5*conf_map.shape[0]/lenth-1.0)
                    criterion3=(peak_src.score+peak_dst.score)*0.1
                    #filter candidate limbs
                    self.debug_print(f"test start:id-{peak_src.idx} pos-{vec_src} end:id-{peak_dst.idx} pos-{vec_dst} c1:{criterion1} c2:{criterion2} c3:{criterion3}")
                    if(criterion1>self.thresh_vec_cnt and criterion2>self.thresh_criterion2):
                        candidate_limbs[limb_idx].append(Connection(peak_src.idx,peak_dst.idx,criterion2+criterion3))
        #filter chosen connection
        all_chosen_limbs=[[] for limb_idx in range(0,len(self.limbs))]
        for limb_idx in range(0,len(self.limbs)):
            sort_candidates=candidate_limbs[limb_idx]
            sort_candidates.sort(reverse=True)
            chosen_limbs=all_chosen_limbs[limb_idx]
            for candidate in sort_candidates:
                assigned=False
                for chosen_limb in chosen_limbs:
                    if(chosen_limb.peak_src_id==candidate.peak_src_id):
                        assigned=True
                    if(chosen_limb.peak_dst_id==candidate.peak_dst_id):
                        assigned=True
                    if(assigned):
                        break
                if(assigned):
                    continue
                chosen_limbs.append(candidate)
        #assemble human
        humans=[]
        for limb_idx,limb in enumerate(self.limbs):
            src_part_idx,dst_part_idx=limb
            chosen_limbs=all_chosen_limbs[limb_idx]
            for chosen_limb in chosen_limbs:
                peak_src_id,peak_dst_id=chosen_limb.peak_src_id,chosen_limb.peak_dst_id
                touched_ids=[]
                for human_id,human in enumerate(humans):
                    if((human[src_part_idx]==peak_src_id) or (human[dst_part_idx]==peak_dst_id)):
                        touched_ids.append(human_id)
                if(len(touched_ids)==1):
                    human=humans[touched_ids[0]]
                    if(human[dst_part_idx]!=peak_dst_id):
                        #TODO: check why could the followers just take the previous larger score???
                        human[dst_part_idx]=peak_dst_id
                        human[19]+=1
                        human[18]+=all_peaks[peak_dst_id].score+chosen_limb.score
                elif(len(touched_ids)>=2):
                    membership=0
                    human_1=humans[touched_ids[0]]
                    human_2=humans[touched_ids[1]]
                    for part_idx in range(0,18):
                        if(human_1[part_idx]>=0 and human_2[part_idx]>=0):
                            membership=2
                    if(membership==0):
                        human_1[0:18]+=human_2[0:18]+1
                        human_1[18]+=human_2[18]+chosen_limb.score
                        human_1[19]+=human_2[19]
                        humans.pop(touched_ids[1])
                    elif(membership==2):
                        human_1[dst_part_idx]=peak_dst_id
                        human_1[19]+=1
                        human_1[18]+=all_peaks[peak_dst_id].score+chosen_limb.score
                elif(len(touched_ids)==0 and limb_idx<17):
                    human=np.zeros(shape=[20]).astype(np.float32)
                    human[:]=-1
                    human[src_part_idx]=peak_src_id
                    human[dst_part_idx]=peak_dst_id
                    human[18]=all_peaks[peak_src_id].score+all_peaks[peak_dst_id].score+chosen_limb.score
                    human[19]=2
                    humans.append(human)
        #return assembled human
        #print(f"test candidate human:{len(humans)}")
        ret_humans=[]
        for human_id,human in enumerate(humans):
            if(self.debug):
                self.debug_print(f"\ntest human filter human_id:{human_id} score:{human[18]/human[19]}  part_num:{human[19]}")
            #print(f"test candidate")
            for i in range(0,18):
                if(human[i]!=-1):
                    peak=all_peaks[int(human[i])]
                    self.debug_print(f"part:{self.parts(i)} loc_y:{peak.y} loc_x:{peak.x} socre:{peak.score}")
            if((human[18]/human[19]>=self.thresh_human_score) and (human[19]>=self.thresh_part_cnt)):
                ret_human=Human(self.parts,self.limbs,self.colors)
                ret_human.local_id=human_id
                ret_human.score=human[18]/human[19]
                for part_idx in range(0,self.n_pos-1):
                    if(human[part_idx]!=-1):
                        peak=all_peaks[int(human[part_idx])]
                        x,y,score=peak.x,peak.y,peak.score
                        ret_human.body_parts[part_idx]=BodyPart(parts=self.parts,u_idx=human[part_idx],part_idx=part_idx,\
                            x=x,y=y,score=score)
                ret_human.global_id=self.cur_id
                ret_humans.append(ret_human)
                self.cur_id+=1
        return ret_humans

    def get_paf_vectors(self,limb_id,vec_src,vec_dst,paf_map):
        def round(x):
            sign_x=np.where(x>0,1,-1)
            return (x+0.5*sign_x).astype(np.int)
        paf_vectors=np.zeros(shape=(self.step_paf,2))
        vec_limb=vec_dst-vec_src
        for step in range(0,self.step_paf):
            vec_loc_y,vec_loc_x=round(vec_src+vec_limb*step/self.step_paf)
            vec_paf_x=paf_map[vec_loc_y][vec_loc_x][limb_id*2]
            vec_paf_y=paf_map[vec_loc_y][vec_loc_x][limb_id*2+1]
            paf_vectors[step][0]=vec_paf_y
            paf_vectors[step][1]=vec_paf_x
        return paf_vectors
    
    def debug_print(self,msg):
        if(self.debug):
            print(msg)
        
class Peak:
    def __init__(self,peak_idx,part_idx,y,x,score):
        self.idx=peak_idx
        self.part_idx=part_idx
        self.y=y
        self.x=x
        self.score=score

class Connection:
    def __init__(self,peak_src_id,peak_dst_id,score):
        self.peak_src_id=peak_src_id
        self.peak_dst_id=peak_dst_id
        self.score=score
    
    def __lt__(self,other):
        return self.score<other.score
    
    def __eq__(self,other):
        return self.score==other.score


