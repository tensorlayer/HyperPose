import os
import cv2
import math
import json
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from ..common import unzip
from .prepare import prepare_dataset
from .visualize import visualize
from .format import CocoMeta,PoseInfo
from .define import CocoPart,CocoColor
from .generate import get_train_dataset, get_eval_dataset

def init_dataset(config):
    dataset=MSCOCO_dataset(config)
    return dataset

class MSCOCO_dataset:
    '''a dataset class specified for coco dataset, provides uniform APIs'''
    def __init__(self,config,input_kpt_cvter=None,output_kpt_cvter=None):
        self.dataset_type=config.data.dataset_type
        self.dataset_version=config.data.dataset_version
        self.dataset_path=config.data.dataset_path
        self.dataset_filter=config.data.dataset_filter
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
        visualize(self.vis_dir,vis_num,train_dataset,self.parts,self.colors)

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

    def get_train_dataset(self):
        '''provide uniform tensorflow dataset for training

        return a tensorflow dataset based on COCO dataset, each iter contains two following object

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
                value: the mask object of the coco dataset, can be docoded in to binary map array by 
                pycocotools.coco.maskUtils.decode() function

            2.3 key: "bbx"
                value: a list of bbx annotation of the image, each bbx is in the [x,y,w,h] form.

        example use
            1.use tensorflow map function to convert the target format
            map_function(image_path,target):

                image = tf.io.read_file(image_path)
                image, target, mask=tf.py_function(defined_pyfunction, [image, target], [tf.float32, tf.float32, tf.float32])

            2.process the target to your own format when in need in defined_pyfunction

            defined_pyfunction(image, target)
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
        return get_train_dataset(self.train_imgs_path,self.train_anns_path,self.dataset_filter,self.input_kpt_cvter)

    def get_eval_dataset(self):
        '''provide uniform tensorflow dataset for evaluating

        return a tensorflow dataset based on COCO dataset, each iter contains two following object

        1.image_path
            a image path string encoded in utf-8 mode
        
        2.image_id
            a image id string encoded in utf-8 mode
        
        example use
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
        return get_eval_dataset(self.val_imgs_path,self.val_anns_path)
    
    def set_input_kpt_cvter(self,input_kpt_cvter):
        self.input_kpt_cvter=input_kpt_cvter
    
    def set_output_kpt_cvter(self,output_kpt_cvter):
        self.output_kpt_cvter=output_kpt_cvter
    
    def get_input_kpt_cvter(self):
        return self.input_kpt_cvter
    
    def get_output_kpt_cvter(self):
        return self.output_kpt_cvter

    def official_eval(self,pd_json,eval_dir=f"./eval_dir"):
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
        pd_anns=pd_json["annotations"]
        for pd_ann in pd_anns:
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

        pd_json_path=f"{eval_dir}/pd_ann.json"
        pd_json_file=open(pd_json_path,"w")
        json.dump(pd_anns,pd_json_file)
        pd_json_file.close()
        #evaluating 
        print(f"evluating on total {len(image_ids)} images...")
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
                print(f"kpst:{gt_ann['keypoints']}")
                print(f"bbxs:{gt_ann['bbox']}")
                print()
        '''

        std_eval=COCOeval(cocoGt=gt_coco,cocoDt=pd_coco,iouType="keypoints")
        std_eval.evaluate()
        std_eval.accumulate()
        std_eval.summarize()