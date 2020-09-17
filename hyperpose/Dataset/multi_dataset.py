import random
import tensorflow as tf
from .base_dataset import Base_dataset
from .common import visualize

class Multi_dataset(Base_dataset):
    def __init__(self,config,combined_dataset_list):
        self.vis_dir=config.data.vis_dir
        self.dataset_type=config.data.dataset_type
        self.combined_dataset_list=combined_dataset_list
        self.parts=combined_dataset_list[0].get_parts()
        self.colors=combined_dataset_list[0].get_colors()

    def visualize(self,vis_num=10):
        train_dataset=self.get_train_dataset()
        visualize(vis_dir=self.vis_dir,vis_num=vis_num,dataset=train_dataset,parts=self.parts,colors=self.colors,\
            dataset_name="multiple_dataset")
    
    def set_parts(self,userdef_parts):
        self.parts=userdef_parts
    
    def set_colors(self,userdef_colors):
        self.colors=userdef_colors

    def get_parts(self):
        return self.parts
    
    def get_colors(self):
        return self.colors
    
    def get_dataset_type(self):
        return self.dataset_type
    
    def generate_train_data(self):
        print("generating training data:")
        train_img_paths_list,train_targets_list=[],[]
        #generate training data individually
        for dataset_idx,dataset in enumerate(self.combined_dataset_list):
            print(f"generating training data from dataset:{dataset_idx} {dataset.dataset_type.name}")
            part_img_paths_list,part_targets_list=dataset.get_train_dataset(in_list=True)
            train_img_paths_list+=part_img_paths_list
            train_targets_list+=part_targets_list
        #shuffle training data
        print("shuffling all combined training data...")
        shuffle_list=[{"image_path":img_path,"target":target} for img_path,target in zip(train_img_paths_list,train_targets_list)]
        random.shuffle(shuffle_list)
        train_img_paths_list=[shuffle_dict["image_path"] for shuffle_dict in shuffle_list]
        train_targets_list=[shuffle_dict["target"] for shuffle_dict in shuffle_list]
        print("shuffling training data finished!")
        print(f"total {len(train_img_paths_list)} combined training data in total generated!")
        return train_img_paths_list,train_targets_list
    
    def generate_eval_data(self):
        print("temply using the evaluate data from the first combined dataset!")
        eval_img_file_list,eval_img_id_list=self.combined_dataset_list[0].generate_eval_data()
        print(f"total {len(eval_img_file_list)} evaluation data in total generated!")
        return eval_img_file_list,eval_img_id_list

    def get_train_dataset(self):
        train_img_paths_list,train_targets_list=self.generate_train_data()
        #tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            assert len(train_img_paths_list) == len(train_targets_list)
            for _input, _target in zip(train_img_paths_list, train_targets_list):
                yield _input.encode('utf-8'), cPickle.dumps(_target)

        train_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))
        return train_dataset

    def get_eval_dataset(self):
        eval_img_file_list,eval_img_id_list=self.generate_eval_data()
        #tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            assert len(eval_img_file_list)==len(eval_img_id_list)
            for img_file,img_id in zip(eval_img_file_list,eval_img_id_list):
                yield img_file.encode("utf-8"),img_id

        eval_dataset = tf.data.Dataset.from_generator(generator,output_types=(tf.string,tf.int32))
        return eval_dataset

    def official_eval(self,pd_json,eval_dir=f"./eval_dir"):
        print("temply using the official_eval from the first combined dataset!")
        return self.combined_dataset_list[0].official_eval(pd_json,eval_dir)