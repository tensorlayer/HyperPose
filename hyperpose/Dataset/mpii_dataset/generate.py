import tensorflow as tf
import tensorlayer as tl
import _pickle as cPickle
from .format import MPIIMeta,PoseInfo

def get_train_dataset(train_images_path,train_annos_path,dataset_filter=None,input_kpt_cvter=lambda x:x):
    #prepare data
    mpii_data=PoseInfo(train_images_path,train_annos_path,dataset_filter=dataset_filter)
    img_paths_list=mpii_data.get_image_list()
    kpts_list=mpii_data.get_kpt_list()
    bbx_list=mpii_data.get_headbbx_list()
    #assemble data
    target_list=[]
    for kpts,bbx in zip(kpts_list,bbx_list):
        for p_idx in range(0,len(kpts)):
            kpts[p_idx]=input_kpt_cvter(kpts[p_idx])
        target_list.append({
            "kpt":kpts,
            "mask":None,
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

def get_eval_dataset(eval_images_path,eval_annos_path,dataset_filter=None):
    #prepare data
    mpii_data=PoseInfo(eval_images_path,eval_annos_path,dataset_filter=dataset_filter)
    img_file_list=mpii_data.get_image_list()
    img_id_list=mpii_data.get_image_id_list()
    #assemble data
    test_img_file_list=img_file_list
    test_img_id_list=img_id_list
    #tensorflow data pipeline
    def generator():
        """TF Dataset generator."""
        assert len(test_img_file_list)==len(test_img_id_list)
        for img_file,img_id in zip(test_img_file_list,test_img_id_list):
            yield img_file.encode("utf-8"),img_id

    test_dataset = tf.data.Dataset.from_generator(generator,output_types=(tf.string,tf.int32))
    return test_dataset
