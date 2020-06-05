import tensorflow as tf
import _pickle as cPickle
from .format import CocoMeta,PoseInfo

def get_pose_data_list(im_path, ann_path,dataset_filter=None):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    print("[x] Get pose data from {}".format(im_path))
    data = PoseInfo(im_path, ann_path, with_mask=True, dataset_filter=dataset_filter)
    imgs_file_list = data.get_image_list()
    objs_info_list = data.get_joint_list()
    mask_list = data.get_mask()
    bbx_list=data.get_bbx_list()
    target_list=[]
    for objs,mask,bbx in zip(objs_info_list,mask_list,bbx_list):
        target_list.append({
            "obj":objs,
            "mask":mask,
            "bbx":bbx
        })
    if len(imgs_file_list) != len(objs_info_list):
        raise Exception("number of images and annotations do not match")
    else:
        print("{} has {} images".format(im_path, len(imgs_file_list)))
    return imgs_file_list, target_list

def get_train_dataset(train_imgs_path,train_anns_path,dataset_filter=None):
    # read coco training images contains valid people
    train_imgs_file_list,train_target_list =get_pose_data_list(train_imgs_path, train_anns_path, dataset_filter=dataset_filter)
    #tensorflow data pipeline
    def generator():
        """TF Dataset generator."""
        assert len(train_imgs_file_list) == len(train_target_list)
        for _input, _target in zip(train_imgs_file_list, train_target_list):
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