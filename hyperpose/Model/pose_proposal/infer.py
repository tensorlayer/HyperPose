import os
import cv2
import json
import numpy as np
from scipy import optimize
from ..human import Human,BodyPart
from .utils import non_maximium_supress

class Post_Processor:
    def __init__(self,parts,limbs,colors,thresh_hold=0.3,eps=1e-8,debug=True):
        self.parts=parts
        self.limbs=limbs
        self.colors=colors
        if(self.colors==None):
            self.colors=[[255,0,0]]*len(self.parts)
        self.n_pos=len(self.parts)
        self.n_limb=len(self.limbs)
        self.eps=eps
        self.cur_id=1
        self.thres_part_score=0.2
        self.thres_edge_score=0.2
        self.thres_nms=0.3
        self.thres_part_cnt=4
        self.thres_human_score=0.1
        self.debug=debug
        self.instance_id=1
        for part_idx in range(0,len(self.parts)):
            if(self.parts(part_idx).name=="Instance"):
                self.instance_id=part_idx
                break
        print(f"PoseProposal Post-processer setting instance id as: {self.instance_id} {self.parts(self.instance_id)}")

    def process(self,pc,pi,px,py,pw,ph,pe,scale_w_rate=1,scale_h_rate=1):
        def get_loc(idx,h,w):
            y=idx//w
            x=idx%w
            return y,x

        #reform output
        _,hout,wout=pc.shape
        L,hnei,wnei,_,_=pe.shape
        bipart_num=hout*wout
        pc=np.clip(pc,0.0,np.inf)
        pi=np.clip(pi,0.0,np.inf)
        pe=np.clip(pe,0.0,np.inf)
        pd_score=(pc).reshape([self.n_pos,bipart_num])
        e_score=np.zeros(shape=(L,bipart_num,bipart_num))
        px=px.reshape([self.n_pos,bipart_num])
        py=py.reshape([self.n_pos,bipart_num])
        pw=pw.reshape([self.n_pos,bipart_num])
        ph=ph.reshape([self.n_pos,bipart_num])
        #construct bbxs
        bbxs_list=[]
        scores_list=[]
        bbxids_list=[]
        assems_list=[]
        for part_idx in range(0,self.n_pos):
            x,y=px[part_idx],py[part_idx]
            w,h=pw[part_idx],ph[part_idx]
            bbxs=np.array([x,y,w,h]).transpose()
            scores=pd_score[part_idx]
            #filte bbxs by score
            filter_ids=np.where(scores>self.thres_part_score)[0]
            filter_bbxs=bbxs[filter_ids]
            filter_scores=scores[filter_ids]

            #non-maximium supress
            left_bbxids=non_maximium_supress(filter_bbxs,filter_scores,self.thres_nms)
            #print(f"test filter_len:{len(filter_ids)} left_len:{len(left_bbxids)}")
            #print(f"test filter_ids:{filter_ids} left_bbxids:{left_bbxids} final_ids:{filter_ids[left_bbxids]}")
            #print(f"test parts:{self.parts(part_idx)} bbx_shape:{filter_bbxs[left_bbxids].shape} score_shape:{filter_scores[left_bbxids].shape}")
            bbxs_list.append(filter_bbxs[left_bbxids])
            scores_list.append(filter_scores[left_bbxids])
            bbxids_list.append(filter_ids[left_bbxids])
            assems_list.append(np.full_like(scores_list[-1],-1))

            #print(f"test nms:\n part:{self.parts(part_idx)}\n chosen_idxs:{bbxids_list[-1]}\n")
        if(self.debug):
            print(f"test bbxs after nms:")
            for part_idx in range(0,self.n_pos):
                bbxs=bbxs_list[part_idx]
                scores=scores_list[part_idx]
                print(f"part:{self.parts(part_idx)},bbx_num:{bbxs.shape[0]}")
                for bbx_id in range(0,bbxs.shape[0]):
                    bbx=bbxs[bbx_id]
                    score=scores[bbx_id]
                    print(f"bbx_id:{bbx_id} x:{bbx[0]*scale_w_rate} y:{bbx[1]*scale_h_rate} w:{bbx[2]*scale_w_rate} h:{bbx[3]*scale_h_rate} score:{score}")
            print()

        #new assemble
        #init egde score
        for l,limb in enumerate(self.limbs):
            for src_id in range(0,bipart_num):
                src_y,src_x=get_loc(src_id,hout,wout)
                for dst_id in range(0,bipart_num):
                    dst_y,dst_x=get_loc(dst_id,hout,wout)
                    delta_y=dst_y-src_y
                    delta_x=dst_x-src_x
                    if((abs(delta_y)>hnei//2) or (abs(delta_x)>wnei//2)):
                        continue 
                    e_score[l][src_id][dst_id]=pe[l][delta_y+hnei//2][delta_x+wnei//2][src_y][src_x]
        e_score=e_score*np.where(e_score>=self.thres_edge_score,1,0)
        #init instance id
        for p_id in range(0,len(bbxs_list[self.instance_id])):
            assems_list[self.instance_id][p_id]=p_id
        #assemble limbs
        for l,limb in enumerate(self.limbs):
            print(f"choosing edge: {self.parts(limb[0])}-{self.parts(limb[1])}:")
            src_part_idx,dst_part_idx=limb
            src_score_list=scores_list[src_part_idx]
            src_bbxid_list=bbxids_list[src_part_idx]
            dst_score_list=scores_list[dst_part_idx]
            dst_bbxid_list=bbxids_list[dst_part_idx]
            match_score=np.zeros(shape=(len(src_score_list),len(dst_score_list)))
            for i,(src_score,src_id) in enumerate(zip(src_score_list,src_bbxid_list)):
                for j,(dst_score,dst_id) in enumerate(zip(dst_score_list,dst_bbxid_list)):
                    match_score[i][j]=src_score*e_score[l][src_id][dst_id]*dst_score
            num_conn=min(len(src_score_list),len(dst_score_list))
            conn_list=[]
            for _ in range(0,num_conn):
                max_score=np.max(match_score)
                if(max_score==0):
                    break
                src_ids, dst_ids=np.nonzero(match_score == max_score)
                conn_list.append((src_ids[0],dst_ids[0],max_score))
                src_id=src_ids[0]
                src_score=src_score_list[src_id]
                src_bbx=bbxs_list[src_part_idx][src_id]
                
                dst_id=dst_ids[0]
                dst_score=dst_score_list[dst_id]
                dst_bbx=bbxs_list[dst_part_idx][dst_id]

                if(self.debug):
                    src_x,src_y,src_w,src_h=src_bbx[0]*scale_w_rate,src_bbx[1]*scale_h_rate,src_bbx[2]*scale_w_rate,src_bbx[3]*scale_h_rate
                    dst_x,dst_y,dst_w,dst_h=dst_bbx[0]*scale_w_rate,dst_bbx[1]*scale_h_rate,dst_bbx[2]*scale_w_rate,dst_bbx[3]*scale_h_rate
                    print(f"chosing edge src_id:{src_id} src_score:{src_score} src_bbx:{src_x},{src_y},{src_w},{src_h}")
                    print(f"chosing edge dst_id:{dst_id} dst_score:{dst_score} dst_bbx:{dst_x},{dst_y},{dst_w},{dst_h}")
                    print()

                match_score[src_ids[0],:]=0
                match_score[:,dst_ids[0]]=0
            for conn in conn_list:
                src_id,dst_id,conn_score=conn
                assems_list[dst_part_idx][dst_id]=assems_list[src_part_idx][src_id]
                #update score
                #scores_list[dst_part_idx][dst_id]=conn_score
            '''
            if(self.debug):
                print(f"test assem list:")
                for part_idx in range(0,self.n_pos):
                    print(f"assems_list {self.parts(part_idx)}:{assems_list[part_idx]}")
            '''
        #assemble humans
        humans=[]
        for _ in range(0,len(bbxs_list[self.instance_id])):
            humans.append(Human(self.parts,self.limbs,self.colors))
        for part_idx in range(0,self.n_pos):
            bbxs,scores,bbx_ids,assem_ids=bbxs_list[part_idx],scores_list[part_idx],bbxids_list[part_idx],assems_list[part_idx]
            for bbx,score,bbx_id,assem_id in zip(bbxs,scores,bbx_ids,assem_ids):
                if(assem_id==-1):
                    continue
                loc_y,loc_x=get_loc(bbx_id,hout,wout)
                x,y,w,h=bbx
                humans[assem_id.astype(np.int)].body_parts[part_idx]=BodyPart(parts=self.parts,u_idx=f"{loc_y}-{loc_x}",part_idx=part_idx,\
                    x=x,y=y,score=score,w=w,h=h)
        filter_humans=[]
        for human in humans:
            if(human.get_partnum()>=self.thres_part_cnt):
                filter_humans.append(human)
        return filter_humans

