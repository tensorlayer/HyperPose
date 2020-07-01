import os
import cv2
import json
import numpy as np
import _pickle as cPickle

from .define import MpiiPart,MpiiColor
from .format import PoseInfo
from .prepare import prepare_dataset
from .visualize import visualize
from .generate import get_train_dataset,get_eval_dataset

def init_dataset(config):
    dataset=MPII_dataset(config)
    return dataset

class MPII_dataset:
    '''a dataset class specified for mpii dataset, provides uniform APIs'''
    def __init__(self,config):
        self.dataset_type=config.data.dataset_type
        self.dataset_path=config.data.dataset_path
        self.dataset_filter=config.data.dataset_filter
        self.vis_dir=config.data.vis_dir
        self.annos_path=None
        self.images_path=None
        self.parts=MpiiPart
        self.colors=MpiiColor
    
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
        visualize(self.vis_dir,vis_num,train_dataset,self.parts,self.colors)
    
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

    def get_train_dataset(self):
        '''provide uniform tensorflow dataset for training

        return a tensorflow dataset based on MPII dataset, each iter contains two following object

        1.image_path
            a image path string encoded in utf-8 mode

        2.target
            bytes of a dict object encoded by _pickle, should be decode by "_pickle.loads(target.numpy())"
            the dict contains the following key-value pair:

            2.1 key: "obj" 
                value: a list of keypoint annotations, each annotation corresponds to a person and is a list of
                keypoints of the person, each keypoint is represent in the [x,y,v] mode, v=0 is unvisible and unanotated,
                v=1 is unvisible but annotated, v=2 is visible and annotated.
            2.2 key: "mask" 
                value: None(MPII doesn't provide any mask information)
            2.3 key: "bbx"
                value: a list of bbx annotation of the image, each bbx is in the [x,y,w,h] form.

        example use
            1.use tensorflow map function to convert the target format
            map_function(image_path,target):

                image = tf.io.read_file(image_path)
                image, target, mask=tf.py_function(defined_pyfunction, [image, target], [tf.float32, tf.float32, tf.float32])
            2.process the target to your own format when in need in defined_pyfunction
            defined_pyfunction(image, target):

                target = _pickle.loads(target.numpy())
                annos = target["obj"]
                mask = target["mask"]
                bbxs = target["bbxs"]
                processing
            3. for image,target in train_dataset  

        for more detail use, one can refer the training pipeline of models.

        Parameters
        ----------
        None

        Returns
        -------
        tensorflow dataset object 
            a unifrom formated tensorflow dataset object for training
        '''
        return get_train_dataset(self.images_path,self.train_annos_path,self.dataset_filter)
    
    def get_eval_dataset(self):
        '''provide uniform tensorflow dataset for evaluating

        return a tensorflow dataset based on MPII dataset, each iter contains two following object

        1.image_path: 
            a image path string encoded in utf-8 mode
        
        2.image_id: 
            a image id string encoded in utf-8 mode
        
        example use:
            for image_path,image_id in eval_dataset

        for more detail use, one can refer the evaluating pipeline of models.

        Parameters
        ----------
        None
        
        Returns
        -------
        tensorflow dataset object 
            a unifrom formated tensorflow dataset object for evaluating
        '''
        return get_eval_dataset(self.images_path,self.val_annos_path,self.dataset_filter)
        
    def official_eval(self,pd_json,eval_dir=f"./eval_dir"):
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
        #format result
        pd_anns=pd_json["annotations"]
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
                gt_ann["keypoints"]=np.concatenate([x,y],axis=0)
            gt_dict[meta.image_id]=gt_ann_list

        all_pd_kpts=[]
        all_gt_kpts=[]
        all_gt_vis=[]
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
                    dist=np.mean(np.linalg.norm((pd_kpts-gt_kpts)*gt_vis,axis=0))
                    if(dist<match_dist):
                        match_dist=dist
                        match_id=gt_id
                if(match_id!=-1):
                    match_pd_ids[match_id]=pd_id
            #add kpts to the list by the matched order 
            for gt_id,gt_img_ann in enumerate(gt_img_anns):
                all_gt_kpts.append(gt_img_ann["keypoints"])
                all_gt_vis.append(gt_img_ann["vis"])
                match_pd_id=match_pd_ids[gt_id]
                if(match_pd_id!=-1):
                    all_pd_kpts.append(pd_img_anns[match_pd_id]["keypoints"])
                #not detected
                else:
                    all_pd_kpts.append(np.zeros_like(all_gt_kpts[-1]))
        #calculate pchk
        #shape kpts 2*n_pos*val_num
        #shape vis n_pos*val_num
        #shape all_dist n_pos*val_num
        #shape headsize val_num
        all_pd_kpts=np.array(all_pd_kpts).transpose([1,2,0])
        all_gt_kpts=np.array(all_gt_kpts).transpose([1,2,0])
        all_gt_vis=np.array(all_gt_vis).transpose([1,0])
        all_gt_headsize=np.linalg.norm(all_gt_kpts[:,MpiiPart.Headtop.value,:]-all_gt_kpts[:,MpiiPart.UpperNeck.value,:],axis=0)
        all_dist=np.linalg.norm(all_pd_kpts-all_gt_kpts,axis=0)/all_gt_headsize
        jnt_vis_num=np.sum(all_gt_vis,axis=1)
        PCKh=100.0*np.sum(all_dist<=0.5,axis=1)/jnt_vis_num
        #calculate pchk_all
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))
        for r in range(0,len(rng)):
            threshold=rng[r]
            pckAll[r]=100.0*np.sum(all_dist<=threshold,axis=1)/jnt_vis_num
        #calculate mean
        PCKh_mask = np.ma.array(PCKh, mask=False)
        PCKh_mask.mask[6:8] = True

        jnt_count = np.ma.array(jnt_vis_num, mask=False)
        jnt_count.mask[6:8] = True
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
            "Mean@0.1": np.sum(pckAll[11,:]*jnt_ratio)
        }
        print("evaluation result-PCKh:")
        for key in result_dict.keys():
            print(f"{key}: {result_dict[key]}")
        result_path=os.path.join(eval_dir,"result.json")
        json.dump(result_dict,open(result_path,"w"))
        return result_dict



    
