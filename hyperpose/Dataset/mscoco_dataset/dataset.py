import os
import cv2
import math
import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..base_dataset import Base_dataset
from ..common import unzip,visualize
from .prepare import prepare_dataset
from .format import CocoMeta,PoseInfo
from .define import CocoPart,CocoColor
from .generate import generate_train_data,generate_eval_data,generate_test_data

def init_dataset(config):
    dataset=MSCOCO_dataset(config)
    return dataset

class MSCOCO_dataset(Base_dataset):
    '''a dataset class specified for coco dataset, provides uniform APIs'''
    def __init__(self,config,input_kpt_cvter=None,output_kpt_cvter=None,dataset_filter=None):
        super().__init__(config,input_kpt_cvter,output_kpt_cvter)
        self.dataset_type=config.data.dataset_type
        self.dataset_version=config.data.dataset_version
        self.dataset_path=config.data.dataset_path
        self.vis_dir=config.data.vis_dir
        self.train_imgs_path,self.train_anns_path=None,None
        self.val_imgs_path,self.val_anns_path=None,None
        self.test_imgs_path,self.test_anns_path=None,None
        self.parts=CocoPart
        self.colors=CocoColor
        if(input_kpt_cvter==None):
            input_kpt_cvter=lambda x:x
        if(output_kpt_cvter==None):
            output_kpt_cvter=lambda x:x
        self.input_kpt_cvter=input_kpt_cvter
        self.output_kpt_cvter=output_kpt_cvter
        self.dataset_filter=dataset_filter
        print(f"using MSCOCO dataset version:{self.dataset_version}")
    
    def visualize(self,vis_num):
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
        visualize(self.vis_dir,vis_num,train_dataset,self.parts,self.colors,dataset_name="mscoco")

    def get_parts(self):
        return self.parts
    
    def get_colors(self):
        return self.colors
    
    def prepare_dataset(self):
        '''download,extract, and reformat the dataset
        the official format is in zip format, extract it into json files and image files. 

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.train_imgs_path,self.train_anns_path,\
            self.val_imgs_path,self.val_anns_path,\
                self.test_imgs_path,self.test_anns_path=prepare_dataset(self.dataset_path,self.dataset_version)
    
    def get_dataset_type(self):
        return self.dataset_type

    def generate_train_data(self):
        return generate_train_data(self.train_imgs_path,self.train_anns_path,self.dataset_filter,self.input_kpt_cvter)
    
    def generate_eval_data(self):
        return generate_eval_data(self.val_imgs_path,self.val_anns_path,self.dataset_filter)
    
    def generate_test_data(self):
        return generate_test_data(self.test_imgs_path,self.test_anns_path)
    
    def set_input_kpt_cvter(self,input_kpt_cvter):
        self.input_kpt_cvter=input_kpt_cvter
    
    def set_output_kpt_cvter(self,output_kpt_cvter):
        self.output_kpt_cvter=output_kpt_cvter
    
    def get_input_kpt_cvter(self):
        return self.input_kpt_cvter
    
    def get_output_kpt_cvter(self):
        return self.output_kpt_cvter

    def official_eval(self,pd_anns,eval_dir=f"./eval_dir"):
        '''providing official evaluation of COCO dataset

        using pycocotool.cocoeval class to perform official evaluation.
        output model metrics of MAPs on coco evaluation dataset

        Parameters
        ----------
        arg1 : String
            A string path of the json file in the same format of cocoeval annotation file(person_keypoints_val2017.json) 
            which contains predicted results. one can refer the evaluation pipeline of models for generation procedure of this json file.
        arg2 : String
            A string path indicates where the json files of filtered intersection part of predict results and ground truth
            the filtered prediction file is stored in eval_dir/pd_ann.json
            the filtered ground truth file is stored in eval_dir/gt_ann.json

        Returns
        -------
        None
        '''

        all_gt_json=json.load(open(self.val_anns_path,"r"))
        all_gt_coco=COCO(self.val_anns_path)
        gt_json={}
        #filter the gt annos
        image_ids=[]
        category_ids=[]
        for pd_idx,pd_ann in enumerate(pd_anns):
            image_ids.append(pd_ann["image_id"])
            category_ids.append(pd_ann["category_id"])
        image_ids=list(np.unique(image_ids))
        category_ids=list(np.unique(category_ids))
        gt_json["info"]=all_gt_json["info"]
        gt_json["licenses"]=all_gt_json["licenses"]
        gt_json["categories"]=all_gt_json["categories"]
        gt_json["images"]=all_gt_coco.loadImgs(ids=image_ids)
        gt_json["annotations"]=all_gt_coco.loadAnns(all_gt_coco.getAnnIds(imgIds=image_ids,catIds=category_ids))

        #save result in json form
        os.makedirs(eval_dir,exist_ok=True)

        gt_json_path=f"{eval_dir}/gt_ann.json"
        gt_json_file=open(gt_json_path,"w")
        json.dump(gt_json,gt_json_file)
        gt_json_file.close()

        pd_json_path=f"{eval_dir}/person_keypoints_val2017_hyperpose_results.json"
        pd_json_file=open(pd_json_path,"w")
        json.dump(pd_anns,pd_json_file)
        pd_json_file.close()
        #evaluating 
        print(f"model predicted evaluation result saved at {pd_json_path}!")
        print(f"evaluating on total {len(image_ids)} images...")
        gt_coco=COCO(gt_json_path)
        pd_coco=gt_coco.loadRes(pd_json_path)

        '''
        #debug
        print(f"test result compare!:")
        for image_id in image_ids:
            print(f"test image_{image_id}:")
            pd_anns=pd_coco.loadAnns(pd_coco.getAnnIds(imgIds=image_id))
            print(f"pd_kpts:{np.array(pd_anns[0]['keypoints']).astype(np.int32)}")
            gt_anns=gt_coco.loadAnns(gt_coco.getAnnIds(imgIds=image_id))
            print(f"gt_kpts:{np.array(gt_anns[0]['keypoints']).astype(np.int32)}")
            
            print(f"test all_info_gt:")
            for gt_ann in gt_anns:
                print(f"kpts:{gt_ann['keypoints']}")
                print(f"bbxs:{gt_ann['bbox']}")
                print()
        '''

        std_eval=COCOeval(cocoGt=gt_coco,cocoDt=pd_coco,iouType="keypoints")
        std_eval.evaluate()
        std_eval.accumulate()
        std_eval.summarize()
    
    def official_test(self,pd_anns,test_dir="./test_dir"):
        server_url="https://competitions.codalab.org/competitions/12061"
        pd_json_path=f"{test_dir}/person_keypoints_test-dev2017_hyperpose_results.json"
        pd_json_file=open(pd_json_path,mode="w")
        json.dump(pd_anns,pd_json_file)
        pd_json_file.close()
        print(f"model predicted test result saved at {pd_json_path}!")
        print(f"please upload the result file to MScoco official server at {server_url} to get official test metrics")