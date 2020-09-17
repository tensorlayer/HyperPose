import os
import glob
import tensorflow as tf

class Imagenet_dataset:
    def __init__(self,config):
        self.dataset_path=config.pretrain.pretrain_dataset_path
        self.train_dataset_path=f"{self.dataset_path}/imagenet/train"
        self.val_dataset_path=f"{self.dataset_path}/imagenet/val"
    
    def prepare_dataset(self):
        # decoding train files
        train_file_names=glob.glob(f"./{self.train_dataset_path}/*.tar")
        print(f"total {len(train_file_names)} tar file for train found!")
        for file_num,file_name in enumerate(train_file_names):
            print(f"decompresssing {file_num+1}/{len(train_file_names)} train tar file...")
            label=file_name[:file_name.rindex(".")]
            os.makedirs(f"{label}",exist_ok=True)
            os.system(f"tar -xf {file_name} -C {label}")
        #decoding val files
        val_file_names=glob.glob(f"./{self.val_dataset_path}/*.tar")
        print(f"total {len(val_file_names)} tar file for evaluation found!")
        for file_num,file_name in enumerate(val_file_names):
            print(f"decompresssing {file_num+1}/{len(val_file_names)} evaluate tar file...")
            label=file_name[:file_name.rindex(".")]
            os.makedirs(f"{label}",exist_ok=True)
            os.system(f"tar -xf {file_name} -C {label}")

    def get_train_dataset(self):
        img_paths=glob.glob(f"{self.train_dataset_path}/*/*")
        if(len(img_paths)==0):
            print(f"error: no training image files founded!")
            print(f"please download the .tar files for training in directory:{self.train_dataset_path} and use Imagenet_dataset.prepare_dataset() to decompress")
            return None
        img_labels={}
        train_img_paths=[]
        train_img_labels=[]
        for img_path in img_paths:
            img_label=os.path.basename(os.path.dirname(img_path))
            if(img_label not in img_labels):
                img_labels[img_label]=len(img_labels)
            train_img_paths.append(img_path)
            train_img_labels.append(img_labels[img_label])
        print(f"total train scenery class num:{len(img_labels)}")
        
        #tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            assert len(train_img_paths)==len(train_img_labels)
            for img_path,img_label in zip(train_img_paths,train_img_labels):
                yield img_path.encode("utf-8"),int(img_label)

        train_dataset=tf.data.Dataset.from_generator(generator,output_types=(tf.string,tf.int64))
        return train_dataset

    def get_eval_dataset(self):
        img_paths=glob.glob(f"{self.val_dataset_path}/*/*")
        if(len(img_paths)==0):
            print(f"error: no evaluate image files founded!")
            print(f"please download the .tar files for evaluate in directory:{self.val_dataset_path} and use Imagenet_dataset.prepare_dataset() to decompress")
            return None
        img_labels={}
        val_img_paths=[]
        val_img_labels=[]
        for img_path in img_paths:
            img_label=os.path.basename(os.path.dirname(img_path))
            if(img_label not in img_labels):
                img_labels[img_label]=len(img_labels)
            val_img_paths.append(img_path)
            val_img_labels.append(img_labels[img_label])
        print(f"total eval scenery class num:{len(img_labels)}")
        
        #tensorflow data pipeline
        def generator():
            """TF Dataset generator."""
            assert len(val_img_paths)==len(val_img_labels)
            for img_path,img_label in zip(val_img_paths,val_img_labels):
                yield img_path.encode("utf-8"),img_label

        val_dataset=tf.data.Dataset.from_generator(generator,output_types=(tf.string,tf.int64))
        return val_dataset