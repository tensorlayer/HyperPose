import os
import cv2
import json
import numpy as np
import _pickle as cPickle

from ..base_dataset import Base_dataset
from ..common import visualize
from .define import MpiiPart,MpiiColor
from .format import PoseInfo
from .prepare import prepare_dataset
from .generate import generate_train_data,generate_eval_data

def init_dataset(config):
    dataset=MPII_dataset(config)
    return dataset

class MPII_dataset(Base_dataset):
    '''a dataset class specified for mpii dataset, provides uniform APIs'''
    def __init__(self,config,input_kpt_cvter=None,output_kpt_cvter=None,dataset_filter=None):
        super().__init__(config,input_kpt_cvter,output_kpt_cvter)
        #basic data configure
        self.official_flag=config.data.official_flag
        self.dataset_type=config.data.dataset_type
        self.dataset_path=config.data.dataset_path
        self.vis_dir=config.data.vis_dir
        self.annos_path=None
        self.images_path=None
        self.parts=MpiiPart
        self.colors=MpiiColor
        if(input_kpt_cvter==None):
            input_kpt_cvter=lambda x:x
        if(output_kpt_cvter==None):
            output_kpt_cvter=lambda x:x
        self.input_kpt_cvter=input_kpt_cvter
        self.output_kpt_cvter=output_kpt_cvter
        self.dataset_filter=dataset_filter
    
    def visualize(self,vis_num=10):
        '''visualize annotations of the train dataset

        visualize the annotation points in the image to help understand and check annotation 
    	the visualized image will be saved in the "data_vis_dir" of the corresponding model directory(specified by model name).
        the visualized annotations are from the train dataset.

        Parameters
        ----------
        arg1 : Int
            An integer indicates how many images with their annotations are going to be visualized.
        
        Returns
        -------
        None
        '''
        train_dataset=self.get_train_dataset()
        visualize(self.vis_dir,vis_num,train_dataset,self.parts,self.colors,dataset_name="mpii")
    
    def get_parts(self):
        return self.parts
    
    def get_colors(self):
        return self.colors
    
    def get_dataset_type(self):
        return self.dataset_type
    
    def prepare_dataset(self):
        '''download,extract, and reformat the dataset
        the official dataset is in .mat format, format it into json format automaticly.

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.train_annos_path,self.val_annos_path,self.images_path=prepare_dataset(self.dataset_path)
    
    def generate_train_data(self):
        return generate_train_data(self.images_path,self.train_annos_path,self.dataset_filter,self.input_kpt_cvter)
    
    def generate_eval_data(self):
        return generate_eval_data(self.images_path,self.val_annos_path,self.dataset_filter)
    
    def generate_test_data(self):
        raise NotImplementedError("MPII test dataset generation has not implemented!")
    
    def set_input_kpt_cvter(self,input_kpt_cvter):
        self.input_kpt_cvter=input_kpt_cvter
    
    def set_output_kpt_cvter(self,output_kpt_cvter):
        self.output_kpt_cvter=output_kpt_cvter

    def get_input_kpt_cvter(self):
        return self.input_kpt_cvter
    
    def get_output_kpt_cvter(self):
        return self.output_kpt_cvter
        
    def official_eval(self,pd_anns,eval_dir=f"./eval_dir"):
        '''providing official evaluation of MPII dataset

        output model metrics of PCHs on mpii evaluation dataset(split automaticly)

        Parameters
        ----------
        arg1 : String
            A string path of the json file in the same format of cocoeval annotation file(person_keypoints_val2017.json) 
            which contains predicted results. one can refer the evaluation pipeline of models for generation procedure of this json file.
        arg2 : String
            A string path indicates where the result json file which contains MPII PCH metrics of various keypoint saves.

        Returns
        -------
        None
        '''
        #format predict result in dict
        pd_dict={}
        for pd_ann in pd_anns:
            image_id=pd_ann["image_id"]
            kpt_list=np.array(pd_ann["keypoints"])
            x=kpt_list[0::3][np.newaxis,...]
            y=kpt_list[1::3][np.newaxis,...]
            pd_ann["keypoints"]=np.concatenate([x,y],axis=0)
            if(image_id not in pd_dict):
                pd_dict[image_id]=[]
            pd_dict[image_id].append(pd_ann)
        #format ground truth
        metas=PoseInfo(self.images_path,self.val_annos_path,dataset_filter=self.dataset_filter).metas
        gt_dict={}
        for meta in metas:
            gt_ann_list=meta.to_anns_list()
            for gt_ann in gt_ann_list:
                kpt_list=np.array(gt_ann["keypoints"])
                x=kpt_list[0::3][np.newaxis,...]
                y=kpt_list[1::3][np.newaxis,...]
                vis_list=np.array(gt_ann["vis"])
                vis_list=np.where(vis_list>0,1,0)
                gt_ann["keypoints"]=np.concatenate([x,y],axis=0)
                gt_ann["vis"]=vis_list
            gt_dict[meta.image_id]=gt_ann_list

        all_pd_kpts=[]
        all_gt_kpts=[]
        all_gt_vis=[]
        all_gt_headbbxs=[]
        #match kpt into order for PCK calculation
        for image_id in pd_dict.keys():
            #sort pd_anns by score
            pd_img_anns=np.array(pd_dict[image_id])
            sort_idx=np.argsort([-pd_img_ann["score"] for pd_img_ann in pd_img_anns])
            pd_img_anns=pd_img_anns[sort_idx]
            gt_img_anns=gt_dict[image_id]
            #start to match pd and gt anns
            match_pd_ids=np.full(shape=len(gt_img_anns),fill_value=-1)
            for pd_id,pd_img_ann in enumerate(pd_img_anns):
                pd_kpts=pd_img_ann["keypoints"]
                match_id=-1
                match_dist=np.inf
                for gt_id,gt_img_ann in enumerate(gt_img_anns):
                    #gt person already matched
                    if(match_pd_ids[gt_id]!=-1):
                        continue
                    gt_kpts=gt_img_ann["keypoints"]
                    gt_vis=gt_img_ann["vis"]
                    vis_mask=np.ones(shape=gt_vis.shape)
                    vis_mask[6:8]=0
                    vis_num=np.sum(gt_vis)
                    if(vis_num==0):
                        continue
                    dist=np.sum(np.linalg.norm((pd_kpts-gt_kpts)*gt_vis*vis_mask,axis=0))/vis_num
                    if(dist<match_dist):
                        match_dist=dist
                        match_id=gt_id
                if(match_id!=-1):
                    match_pd_ids[match_id]=pd_id
            #add kpts to the list by the matched order 
            for gt_id,gt_img_ann in enumerate(gt_img_anns):
                all_gt_kpts.append(gt_img_ann["keypoints"])
                all_gt_vis.append(gt_img_ann["vis"])
                all_gt_headbbxs.append(gt_img_ann["headbbx"])
                match_pd_id=match_pd_ids[gt_id]
                if(match_pd_id!=-1):
                    all_pd_kpts.append(pd_img_anns[match_pd_id]["keypoints"])
                #not detected
                else:
                    all_pd_kpts.append(np.zeros_like(all_gt_kpts[-1]))
        #calculate pchk
        #input shape:
        #shape kpts 2*n_pos*val_num
        #shape vis n_pos*val_num
        #shape headbbxs(x,y,w,h) 4*val_num
        #shape all_dist n_pos*val_num
        #shape headsize val_num
        print(f"evaluating over {len(pd_dict.keys())} images and {len(all_gt_kpts)} people")
        all_pd_kpts=np.array(all_pd_kpts).transpose([1,2,0])
        all_gt_kpts=np.array(all_gt_kpts).transpose([1,2,0])
        all_gt_vis=np.array(all_gt_vis).transpose([1,0])
        all_gt_headbbxs=np.array(all_gt_headbbxs).transpose([1,0])
        all_gt_headsize=np.linalg.norm(all_gt_headbbxs[2:4,:],axis=0) #[2:4] correspond to w,h
        all_dist=np.linalg.norm(all_pd_kpts-all_gt_kpts,axis=0)/all_gt_headsize
        jnt_vis_num=np.sum(all_gt_vis,axis=1)
        PCKh=100.0*np.sum(all_dist<=0.5,axis=1)/jnt_vis_num
        #calculate pchk_all
        rng = np.arange(0, 0.5+0.1, 0.1)
        pckAll = np.zeros((len(rng), len(self.parts)))
        for r in range(0,len(rng)):
            threshold=rng[r]
            pckAll[r]=100.0*np.sum(all_dist<=threshold,axis=1)/jnt_vis_num
        #calculate mean
        PCKh_mask = np.ma.array(PCKh, mask=False)
        PCKh_mask.mask[6:8] = True      #ignore thorax and pevis

        jnt_count = np.ma.array(jnt_vis_num, mask=False)
        jnt_count.mask[6:8] = True      #ignore thorax and pevis
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
        
        result_dict={
            "Head":     PCKh[MpiiPart.Headtop.value],
            "Shoulder": 0.5*(PCKh[MpiiPart.LShoulder.value]+PCKh[MpiiPart.RShoulder.value]),
            "Elbow":    0.5*(PCKh[MpiiPart.LElbow.value]+PCKh[MpiiPart.RElbow.value]),
            "Wrist":    0.5*(PCKh[MpiiPart.LWrist.value]+PCKh[MpiiPart.RWrist.value]),
            "Hip":      0.5*(PCKh[MpiiPart.LHip.value]+PCKh[MpiiPart.RHip.value]),
            "Knee":     0.5*(PCKh[MpiiPart.LKnee.value]+PCKh[MpiiPart.RKnee.value]),
            "Ankle":    0.5*(PCKh[MpiiPart.LAnkle.value]+PCKh[MpiiPart.RAnkle.value]),
            "Mean":     np.sum(PCKh_mask*jnt_ratio),
            "Mean@0.1": np.mean(np.sum(pckAll[1:,:]*jnt_ratio,axis=1))
        }
        print("\tresult-PCKh:")
        for key in result_dict.keys():
            print(f"\t{key}:   {result_dict[key]}")
        result_path=os.path.join(eval_dir,"result.json")
        json.dump(result_dict,open(result_path,"w"))
        return result_dict

    def official_test(self,pd_anns,test_dir="./test_dir"):
        raise NotImplementedError("test over MPII dataset haven't implemented yet!")
    
