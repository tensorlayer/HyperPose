import tensorflow as tf
import _pickle as cPickle
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
    # read coco training images contains valid people
    coco_data=PoseInfo(val_imgs_path,val_anns_path,with_mask=False, dataset_filter=dataset_filter, eval=True)
    img_file_list,img_id_list=coco_data.get_image_list(),coco_data.get_image_id_list()
    print(f"test gen_eval_data:")
    print(f"test len img_file_list:{len(img_file_list)} img_id_list:{len(img_id_list)}")
    return img_file_list,img_id_list