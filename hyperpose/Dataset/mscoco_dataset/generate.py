import os
import tensorflow as tf
import _pickle as cPickle
from pycocotools.coco import COCO
from .format import CocoMeta,PoseInfo

def generate_train_data(train_imgs_path,train_anns_path,dataset_filter=None,input_kpt_cvter=lambda x: x):
    # read coco training images contains valid people
    data = PoseInfo(train_imgs_path, train_anns_path, with_mask=True, dataset_filter=dataset_filter)
    img_paths_list = data.get_image_list()
    kpts_list = data.get_kpt_list()
    mask_list = data.get_mask_list()
    bbx_list=data.get_bbx_list()
    target_list=[]
    for kpts,mask,bbx in zip(kpts_list,mask_list,bbx_list):
        target_list.append({
            "kpt":kpts,
            "mask":mask,
            "bbx":bbx,
            "labeled":1
        })
    return img_paths_list,target_list

def generate_eval_data(val_imgs_path,val_anns_path,dataset_filter=None):
    # read coco evaluation images contains valid people
    coco_data=PoseInfo(val_imgs_path,val_anns_path,with_mask=False, dataset_filter=dataset_filter, eval=True)
    img_file_list,img_id_list=coco_data.get_image_list(),coco_data.get_image_id_list()
    return img_file_list,img_id_list

def generate_test_data(test_imgs_path,test_anns_path):
    # read coco test-dev images used for test
    print("currently using the test-dev dataset for test! if you want to test over the whole test2017 dataset, change the annotation path please!")
    dev_coco=COCO(test_anns_path)
    img_id_list=dev_coco.getImgIds()
    img_file_list=[]
    for img_id in img_id_list:
        img_info=dev_coco.loadImgs(img_id)[0]
        img_file=img_info["file_name"]
        img_path=os.path.join(test_imgs_path,img_file)
        img_file_list.append(img_path)
    return img_file_list,img_id_list
