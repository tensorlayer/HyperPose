import numpy as np
import tensorflow as tf
import tensorlayer as tl
import _pickle as cPickle
from .format import MPIIMeta,PoseInfo

def generate_train_data(train_images_path,train_annos_path,dataset_filter=None,input_kpt_cvter=lambda x:x):
    #prepare data
    mpii_data=PoseInfo(train_images_path,train_annos_path,dataset_filter=dataset_filter)
    img_paths_list=mpii_data.get_image_list()
    kpts_list=mpii_data.get_kpt_list()
    bbx_list=mpii_data.get_headbbx_list()
    #assemble data
    target_list=[]
    for kpts,head_bbx in zip(kpts_list,bbx_list):
        bbx=np.array(head_bbx).copy()
        bbx[:,2]=bbx[:,2]*4
        bbx[:,3]=bbx[:,3]*4
        target_list.append({
            "kpt":kpts,
            "mask":None,
            "bbx":bbx,
            "head_bbx":head_bbx,
            "labeled":1
        })
    return img_paths_list,target_list

def generate_eval_data(eval_images_path,eval_annos_path,dataset_filter=None):
    #prepare data
    mpii_data=PoseInfo(eval_images_path,eval_annos_path,dataset_filter=dataset_filter)
    img_file_list=mpii_data.get_image_list()
    img_id_list=mpii_data.get_image_id_list()
    return img_file_list,img_id_list

def generate_test_data(test_images_path,test_annos_path):
    raise NotImplementedError("MPII test dataset generation has not implemented!")
