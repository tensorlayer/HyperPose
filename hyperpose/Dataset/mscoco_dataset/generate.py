import tensorflow as tf
import _pickle as cPickle
from .format import CocoMeta,PoseInfo

def get_train_dataset(train_imgs_path,train_anns_path,dataset_filter=None,input_kpt_cvter=lambda x: x):
    # read coco training images contains valid people
    data = PoseInfo(train_imgs_path, train_anns_path, with_mask=True, dataset_filter=dataset_filter)
    img_paths_list = data.get_image_list()
    kpts_list = data.get_kpt_list()
    mask_list = data.get_mask_list()
    bbx_list=data.get_bbx_list()
    target_list=[]
    for kpts,mask,bbx in zip(kpts_list,mask_list,bbx_list):
        for p_idx in range(0,len(kpts)):
            kpts[p_idx]=input_kpt_cvter(kpts[p_idx])
        target_list.append({
            "kpt":kpts,
            "mask":mask,
            "bbx":bbx
        })
    train_img_paths_list=img_paths_list
    train_target_list=target_list
    
    #tensorflow data pipeline
    def generator():
        """TF Dataset generator."""
        assert len(train_img_paths_list) == len(train_target_list)
        for _input, _target in zip(train_img_paths_list, train_target_list):
            yield _input.encode('utf-8'), cPickle.dumps(_target)

    train_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))
    return train_dataset

def get_eval_dataset(val_imgs_path,val_anns_path,dataset_filter=None):
    # read coco training images contains valid people
    coco_data=PoseInfo(val_imgs_path,val_anns_path,with_mask=False, dataset_filter=dataset_filter, eval=True)
    img_file_list,img_id_list=coco_data.get_image_list(),coco_data.get_image_id_list()
    #tensorflow data pipeline
    def generator():
        """TF Dataset generator."""
        assert len(img_id_list)==len(img_file_list)
        for img_file,img_id in zip(img_file_list,img_id_list):
            yield img_file.encode("utf-8"),img_id

    eval_dataset = tf.data.Dataset.from_generator(generator,output_types=(tf.string,tf.int32))
    return eval_dataset