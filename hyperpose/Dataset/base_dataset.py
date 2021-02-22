import os
import random
import numpy as np
import _pickle as cPickle
import tensorflow as tf
from .common import DATA,get_domainadapt_targets

class Base_dataset:
    def __init__(self,config,input_kpt_cvter=lambda x: x,output_kpt_cvter=lambda x: x):
        # dataset basic configure
        self.official_flag=config.data.official_flag
        self.dataset_type=config.data.dataset_type
        self.dataset_path=config.data.dataset_path
        self.dataset_filter=config.data.dataset_filter
        self.input_kpt_cvter=input_kpt_cvter
        self.output_kpt_cvter=output_kpt_cvter
        self.train_datasize=0
        self.eval_datasize=0
        self.test_datasize=0
        # user-define dataset configure
        self.useradd_flag=config.data.useradd_flag
        self.useradd_scale_rate=config.data.useradd_scale_rate
        self.useradd_train_img_paths=config.data.useradd_train_img_paths
        self.useradd_train_targets=config.data.useradd_train_targets
        # domain adaptation
        self.domainadapt_flag=config.data.domainadapt_flag
        self.domainadapt_scale_rate=config.data.domainadapt_scale_rate
        self.domainadapt_train_img_paths=config.data.domainadapt_train_img_paths
        self.domainadapt_train_targets=get_domainadapt_targets(self.domainadapt_train_img_paths)

    def visualize(self,vis_num=10):
        raise NotImplementedError("virtual class Base_dataset function: visualize not implemented!")

    def set_dataset_version(self,dataset_version):
        self.dataset_version=self.dataset_version   
    
    def get_parts(self):
        raise NotImplementedError("virtual class Base_dataset function: get_parts not implemented!")
    
    def get_colors(self):
        raise NotImplementedError("virtual class Base_dataset function: get_colors not implemented!")
    
    def generate_train_data(self):
        raise NotImplementedError("virtual class Base_dataset function: get_train_dataset not implemented!")
    
    def generate_eval_data(self):
        raise NotImplementedError("virtual class Base_dataset function: get_eval_dataset not implemented!")
    
    def get_dataset_type(self):
        return DATA.USERDEF
    
    def get_train_datasize(self):
        #make sure cal this API to get datasize after calling get_train_dataset
        return self.train_datasize
    
    def get_eval_datasize(self):
        #make sure cal this API to get datasize after calling get_eval_dataset
        return self.eval_datasize
    
    def get_test_datasize(self):
        #make sure cal this API to get datasize after calling get_test_dataset
        return self.test_datasize

    def get_train_dataset(self,in_list=False,need_total_num=False):
        '''provide uniform tensorflow dataset for training

        return a tensorflow dataset based on MPII dataset, each iter contains two following object

        1.image_path
            a image path string encoded in utf-8 mode

        2.target
            bytes of a dict object encoded by _pickle, should be decode by "_pickle.loads(target.numpy())"
            the dict contains the following key-value pair:

            2.1 key: "kpt" 
                value: a list of keypoint annotations, each annotation corresponds to a person and is a list of
                keypoints of the person, each keypoint is represent in the [x,y,v] mode, v=0 is unvisible and unanotated,
                v=1 is unvisible but annotated, v=2 is visible and annotated.
            2.2 key: "mask" 
                value: None(MPII doesn't provide any mask information)
            2.3 key: "bbx"
                value: a list of bbx annotation of the image, each bbx is in the [x,y,w,h] form.
            2.4 key: "labeled"(optional)
                value: a bool value used for damain adaptation, 0 stands for the unlabeled target domain, 1 stands for the labeled src domain

        example use
            1.use tensorflow map function to convert the target format
            map_function(image_path,target):

                image = tf.io.read_file(image_path)
                image, target, mask=tf.py_function(defined_pyfunction, [image, target], [tf.float32, tf.float32, tf.float32])
            2.process the target to your own format when in need in defined_pyfunction
            defined_pyfunction(image, target):

                target = _pickle.loads(target.numpy())
                annos = target["kpt"]
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
        train_img_paths_list,train_targets_list=[],[]
        #official data
        if(self.official_flag):
            print("generating official training data...")
            official_img_paths_list,official_targets_list=self.generate_train_data()
            assert len(official_img_paths_list)==len(official_targets_list)
            train_img_paths_list+=official_img_paths_list
            train_targets_list+=official_targets_list
            print(f"{len(train_img_paths_list)} official training data added!")
        #user defined data
        if(self.useradd_flag):
            print("adding user defined training data...")
            assert len(self.useradd_train_img_paths)==len(self.useradd_train_targets)
            train_img_paths_list+=self.useradd_train_img_paths*self.useradd_scale_rate
            train_targets_list+=self.useradd_train_targets*self.useradd_scale_rate
            print(f"{len(self.useradd_train_img_paths)} user define training data added! repeat time:{self.useradd_scale_rate}")
        #domain adaptation data
        if(self.domainadapt_flag):
            print("adding domain adaptation training data...")
            assert len(self.domainadapt_train_img_paths==len(self.domainadapt_train_targets))
            train_img_paths_list+=self.domainadapt_train_img_paths*self.domainadapt_scale_rate
            train_targets_list+=self.domainadapt_train_targets*self.domainadapt_scale_rate
            print(f"{len(self.domainadapt_train_img_paths)} domain adaptation data added! repeat time:{self.domainadapt_scale_rate}")
        #filter non-exist image and target
        print("filtering non-exist images and targets")
        filter_train_img_paths,filter_train_targets=[],[]
        filter_num=0
        for train_img_path,train_target in zip(train_img_paths_list,train_targets_list):
            if(os.path.exists(train_img_path)):
                filter_train_img_paths.append(train_img_path)
                filter_train_targets.append(train_target)
            else:
                filter_num+=1
        train_img_paths_list=filter_train_img_paths
        train_targets_list=filter_train_targets
        print(f"filtering finished! total {len(train_img_paths_list)} images and targets left, {filter_num} invalid found.")
        #input conversion
        print("converting input keypoint...")
        for target_idx in range(0,len(train_targets_list)):
            target=train_targets_list[target_idx]
            #keypoint conversion
            kpts=target["kpt"]
            for p_idx in range(0,len(kpts)):
                kpts[p_idx]=self.input_kpt_cvter(np.array(kpts[p_idx]))
            target["kpt"]=kpts
            train_targets_list[target_idx]=target
        print("conversion finished!")
        #shuffle all data
        print("shuffling all training data...")
        shuffle_list=[{"image_path":img_path,"target":target} for img_path,target in zip(train_img_paths_list,train_targets_list)]
        random.shuffle(shuffle_list)
        train_img_paths_list=[shuffle_dict["image_path"] for shuffle_dict in shuffle_list]
        train_targets_list=[shuffle_dict["target"] for shuffle_dict in shuffle_list]
        print("shuffling data finished, generating tensorflow dataset...")
        print(f"total {len(train_img_paths_list)} training data generated!")

        #tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            assert len(train_img_paths_list) == len(train_targets_list)
            for _input, _target in zip(train_img_paths_list, train_targets_list):
                yield _input.encode('utf-8'), cPickle.dumps(_target)
        
        train_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))
        #update datasize
        self.train_datasize=len(train_img_paths_list)
        print(f"train dataset generation finished!")
        if(in_list):
            return train_img_paths_list,train_targets_list
        else:
            return train_dataset
    
    def get_eval_dataset(self,in_list=False):
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
        print("generating official evaluating data...")
        eval_img_files_list,eval_img_ids_list=self.generate_eval_data()
        print(f"total {len(eval_img_files_list)} official evaluating data generated!")
        #filter non-exist eval images and targets
        print("filtering non-exist images and targets")
        filter_img_files,filter_img_ids=[],[]
        filter_num=0
        for img_file,img_id in zip(eval_img_files_list,eval_img_ids_list):
            if(os.path.exists(img_file)):
                filter_img_files.append(img_file)
                filter_img_ids.append(img_id)
            else:
                filter_num+=1
        eval_img_files_list=filter_img_files
        eval_img_ids_list=filter_img_ids
        print(f"filtering finished! total {len(eval_img_files_list)} images and targets left, {filter_num} invalid found.")
        #tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            assert len(eval_img_files_list)==len(eval_img_ids_list)
            for img_file,img_id in zip(eval_img_files_list,eval_img_ids_list):
                yield img_file.encode("utf-8"),img_id

        eval_dataset = tf.data.Dataset.from_generator(generator,output_types=(tf.string,tf.int32))
        #update datasize
        self.eval_datasize=len(eval_img_files_list)
        print(f"eval dataset generation finished!")
        if(in_list):
            return eval_img_files_list,eval_img_ids_list
        else:
            return eval_dataset
    
    def get_test_dataset(self,in_list=False):
        print("generating official test dataset...")
        test_img_files_list,test_img_ids_list=self.generate_test_data()
        print(f"total {len(test_img_files_list)} official test data generated!")
        #filter non-exist test images and targets
        filter_img_files_list,filter_img_ids_list=[],[]
        filter_num=0
        for img_file,img_id in zip(test_img_files_list,test_img_ids_list):
            if(os.path.exists(img_file)):
                filter_img_files_list.append(img_file)
                filter_img_ids_list.append(img_id)
            else:
                filter_num+=1
        test_img_files_list=filter_img_files_list
        test_img_ids_list=filter_img_ids_list
        print(f"filtering finished! total {len(test_img_files_list)} images and targets left, {filter_num} invalid found.")
        #tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            assert len(test_img_files_list)==len(test_img_ids_list)
            for img_file,img_id in zip(test_img_files_list,test_img_ids_list):
                yield img_file.encode("utf-8"),img_id
        
        test_dataset=tf.data.Dataset.from_generator(generator,output_types=(tf.string,tf.int32))
        #update datasize
        self.test_datasize=len(test_img_files_list)
        print("test dataset generation finished!")
        if(in_list):
            return test_img_files_list,test_img_ids_list
        else:
            return test_dataset

    def official_eval(self,pd_json,eval_dir=f"./eval_dir"):
        raise NotImplementedError("virtual class Base_dataset function: official_eval not implemented!")
    
    def set_input_kpt_cvter(self,input_kpt_cvter):
        self.input_kpt_cvter=input_kpt_cvter
    
    def set_output_kpt_cvter(self,output_kpt_cvter):
        self.output_kpt_cvter=output_kpt_cvter

    def get_input_kpt_cvter(self):
        return self.input_kpt_cvter
    
    def get_output_kpt_cvter(self):
        return self.output_kpt_cvter
        