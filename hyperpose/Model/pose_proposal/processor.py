import os
import numpy as np
from .utils import non_maximium_supress
from .utils import get_pose_proposals
from .utils import draw_bbx, draw_edge
from ..human import Human,BodyPart
from ..processor import BasicPreProcessor
from ..processor import BasicPostProcessor
from ..processor import BasicVisualizer
from ..processor import PltDrawer
from ..common import to_numpy_dict, image_float_to_uint8

class PreProcessor(BasicPreProcessor):
    def __init__(self,parts,limbs,hin,win,hout,wout,hnei,wnei,colors=None,data_format="channels_first",*args,**kargs):
        self.hin=hin
        self.win=win
        self.hout=hout
        self.wout=wout
        self.hnei=hnei
        self.wnei=wnei
        self.parts=parts
        self.limbs=limbs
        self.data_format=data_format
        self.colors=colors if (colors!=None) else (len(self.parts)*[[0,255,0]])
    
    def process(self, annos, mask, bbxs):
        gc,gx,gy,gw,gh,ge,ge_mask=get_pose_proposals(annos,bbxs,self.hin,self.win,self.hout,self.wout,self.hnei,self.wnei,\
            self.parts,self.limbs,mask,self.data_format)
        target_x = {"c":gc, "x":gx, "y":gy, "w":gw, "h":gh, "e":ge, "e_mask":ge_mask}
        return target_x

class PostProcessor(BasicPostProcessor):
    def __init__(self,parts,limbs,thresh_hold=0.3,eps=1e-8,colors=None,debug=False,*args,**kargs):
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

    def process(self, predict_x, scale_w_rate=1,scale_h_rate=1, resize=True):
        predict_x = to_numpy_dict(predict_x)
        batch_size = list(predict_x.values())[0].shape[0]
        humans_list = []
        for batch_idx in range(0,batch_size):
            predict_x_one = {key:value[batch_idx] for key,value in predict_x.items()}
            humans_list.append(self.process_one(predict_x_one, scale_w_rate, scale_h_rate, resize=resize))        
        return humans_list

    def process_one(self,predict_x,scale_w_rate=1,scale_h_rate=1, resize=True):
        pc, px, py, pw, ph, pi, pe  = predict_x["c"], predict_x["x"], predict_x["y"], predict_x["w"], predict_x["h"],\
                                            predict_x["i"], predict_x["e"]
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

class Visualizer(BasicVisualizer):
    def __init__(self, parts, limbs, save_dir="./save_dir", *args, **kargs):
        self.parts = parts
        self.limbs = limbs
        self.save_dir = save_dir
    
    def visualize(self, image_batch, predict_x, mask_batch=None, humans_list=None, name="vis"):
        # mask
        if(mask_batch is None):
            mask_batch = np.ones_like(image_batch)
        # transform
        image_batch = np.transpose(image_batch,[0,2,3,1])
        mask_batch = np.transpose(mask_batch,[0,2,3,1])
        # defualt values
        # TODO: pass config values
        threshhold=0.3
        hout, wout= 12, 12
        wnei, hnei= 9, 9

        # predict maps
        pc_batch, px_batch, py_batch, pw_batch, ph_batch, pi_batch, pe_batch  = predict_x["c"], predict_x["x"], predict_x["y"], \
                                                            predict_x["w"], predict_x["h"], predict_x["i"], predict_x["e"]

        # draw figures
        batch_size = image_batch.shape[0]
        for b_idx in range(0,batch_size):
            image, mask = image_batch[b_idx], mask_batch[b_idx]
            pc, px, py, pw, ph, pi, pe = pc_batch[b_idx], px_batch[b_idx], py_batch[b_idx], pw_batch[b_idx], ph_batch[b_idx],\
                                                                pi_batch[b_idx], pe_batch[b_idx]

            # begin draw
            pltdrawer = PltDrawer(draw_row=1, draw_col=2)
            
            # draw original image
            origin_image = image.copy()
            origin_image = image_float_to_uint8(origin_image)
            pltdrawer.add_subplot(origin_image, "origin image")

            # draw predict image
            pd_image = origin_image.copy()
            pd_image = draw_bbx(pd_image,pc,px,py,pw,ph,threshhold)
            pd_image = draw_edge(pd_image,pe,px,py,pw,ph,hnei,wnei,hout,wout,self.limbs,threshhold)
            pltdrawer.add_subplot(pd_image, "predict image")

            # save figure
            pltdrawer.savefig(f"{self.save_dir}/{name}_{b_idx}.png")

            # draw results
            if(humans_list is not None):
                humans = humans_list[b_idx]
                self.visualize_result(image, humans, name=f"{name}_{b_idx}_result")
        

    def visualize_compare(self, image_batch, predict_x, target_x, mask_batch=None, humans_list=None, name="vis"):
        # mask
        if(mask_batch is None):
            mask_batch = np.ones_like(image_batch)
        # transform
        image_batch = np.transpose(image_batch,[0,2,3,1])
        mask_batch = np.transpose(mask_batch,[0,2,3,1])
        # defualt values
        # TODO: pass config values
        threshhold = 0.3
        hout, wout= 12, 12
        wnei, hnei= 9, 9

        # predict maps
        pc_batch, px_batch, py_batch, pw_batch, ph_batch, pi_batch, pe_batch  = predict_x["c"], predict_x["x"], predict_x["y"], \
                                                            predict_x["w"], predict_x["h"], predict_x["i"], predict_x["e"]
        # target maps
        gc_batch, gx_batch, gy_batch, gw_batch, gh_batch, ge_mask_batch, ge_batch  = target_x["c"], target_x["x"], target_x["y"], \
                                                            target_x["w"], target_x["h"], target_x["i"], target_x["e"]
        
        # draw figures
        batch_size = image_batch.shape[0]
        for b_idx in range(0,batch_size):
            image, mask = image_batch[b_idx], mask_batch[b_idx]
            pc, px, py, pw, ph, pi, pe = pc_batch[b_idx], px_batch[b_idx], py_batch[b_idx], pw_batch[b_idx], ph_batch[b_idx],\
                                                                pi_batch[b_idx], pe_batch[b_idx]
            gc, gx, gy, gw, gh, ge_mask, ge = gc_batch[b_idx], gx_batch[b_idx], gy_batch[b_idx], gw_batch[b_idx], gh_batch[b_idx],\
                                                                ge_mask_batch[b_idx], ge_batch[b_idx]

            # begin draw
            pltdrawer = PltDrawer(draw_row=2, draw_col=2)
            
            # draw original image
            origin_image = image_float_to_uint8(image.copy())
            pltdrawer.add_subplot(origin_image, "origin image")

            # draw mask
            pltdrawer.add_subplot(mask, "mask")

            # draw predict image
            pd_image = origin_image.copy()
            pd_image = draw_bbx(pd_image,pc,px,py,pw,ph,threshhold)
            pd_image = draw_edge(pd_image,pe,px,py,pw,ph,hnei,wnei,hout,wout,self.limbs,threshhold)
            pltdrawer.add_subplot(pd_image, "predict image")
            
            # draw ground truth image
            gt_image = origin_image.copy()
            gt_image=draw_bbx(gt_image,gc,gx,gy,gw,gh,threshhold)
            gt_image=draw_edge(gt_image,ge,gx,gy,gw,gh,hnei,wnei,hout,wout,self.limbs,threshhold)
            pltdrawer.add_subplot(gt_image, "groundtruth image")

            # save figure
            pltdrawer.savefig(f"{self.save_dir}/{name}_{b_idx}.png")

            # draw results
            if(humans_list is not None):
                humans = humans_list[b_idx]
                self.visualize_result(image, humans, name=f"{name}_{b_idx}_result")
        

