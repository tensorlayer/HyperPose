
import os
import json
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files.utils import (del_file, folder_exists, maybe_download_and_extract)
from ..common import unzip
from .format import generate_json

def prepare_dataset(dataset_path):
    path=os.path.join(dataset_path,"mpii")
    #prepare annotation
    annos_dir=os.path.join(path,"mpii_human_pose_v1_u12_2")
    mat_annos_path=os.path.join(annos_dir,"mpii_human_pose_v1_u12_1.mat")
    if(os.path.exists(mat_annos_path) is False):
        logging.info("    downloading annotations")
        os.system(f"wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip -P {path}")
        unzip(os.path.join(path,"mpii_human_pose_v1_u12_2.zip"),path)
        del_file(os.path.join(path,"mpii_human_pose_v1_u12_2.zip"))
    else:
        logging.info("    annotations exist")
    #prepare json annotation
    train_annos_path=os.path.join(annos_dir,"mpii_human_pose_train.json")
    val_annos_path=os.path.join(annos_dir,"mpii_human_pose_val.json")
    if((not os.path.exists(train_annos_path)) or (not os.path.exists(val_annos_path))):
        print("    json annotation doesn't exist, generaing json annotations...")
        json_annos=generate_json(mat_annos_path)
        #left 3000 images for val
        split_val_ids=None
        split_path=os.path.join(annos_dir,"split_val.json")
        if(os.path.exists(split_path)):
            print("using preset split val set")
            split_val_ids=json.load(open(split_path,"r"))
        else:
            print("using the first 3000 annotations as default val dataset")
        train_annos={}
        val_annos={}
        for annos_num,image_path in enumerate(json_annos.keys()):
            if(split_val_ids!=None):
                if(image_path in split_val_ids):
                    val_annos[image_path]=json_annos[image_path]
                else:
                    train_annos[image_path]=json_annos[image_path]
            else:
                if(annos_num<3000):
                    val_annos[image_path]=json_annos[image_path]
                else:
                    train_annos[image_path]=json_annos[image_path]
        print(f"generated train annos:{len(train_annos.keys())}")
        print(f"generated val annos:{len(val_annos.keys())}")
        train_file=open(train_annos_path,"w")
        json.dump(train_annos,train_file)
        train_file.close()
        val_file=open(val_annos_path,"w")
        json.dump(val_annos,val_file)
        val_file.close()
        print("    json annotation generation finished!")

    #prepare image
    images_path=os.path.join(path,"images")
    if(os.path.exists(images_path) is False):
        logging.info("    downloading images")
        os.system(f"wget https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz -P {path}")
        os.system(f"tar -xvf {os.path.join(path,'mpii_human_pose_v1.tar.gz')}")
        del_file(os.path.join(path,"mpii_human_pose_v1.tar.gz"))
    else:
        logging.info("    images exist")
    return train_annos_path,val_annos_path,images_path