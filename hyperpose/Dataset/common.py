import logging
import os
import cv2
import numpy as np
import zipfile
from enum import Enum
import _pickle as cpickle
import tensorflow as tf
import matplotlib.pyplot as plt
from ..Config.define import TRAIN,MODEL,DATA,KUNGFU
import multiprocessing

def unzip(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

def imread_rgb_float(image_path,data_format="channels_first"):
    image=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    if(data_format=="channels_first"):
        image=np.transpose(image,[2,0,1])
    return image.copy()

def imwrite_rgb_float(image,image_path,data_format="channels_first"):
    if(data_format=="channels_first"):
        image=np.transpose(image,[1,2,0])
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image=np.clip(image*255.0,0,255).astype(np.uint8)
    return cv2.imwrite(image_path,image)

def file_log(log_file,msg):
    log_file.write(msg+"\n")
    print(msg)

def visualize(vis_dir,vis_num,dataset,parts,colors,dataset_name="default"):
    log_file=open(os.path.join(vis_dir,"visualize_info.txt"),mode="w")
    for vis_id,(img_file,annos) in enumerate(dataset,start=1):
        if(vis_id>=vis_num):
            break
        image=cv2.cvtColor(cv2.imread(img_file.numpy().decode("utf-8"),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        radius=int(np.round(min(image.shape[0],image.shape[1])/40))
        annos=cpickle.loads(annos.numpy())
        ori_img=image
        vis_img=image.copy()
        kpts_list=annos["kpt"]
        file_log(log_file,f"visualizing image:{img_file} with {len(kpts_list)} humans...")
        for pid,kpts in enumerate(kpts_list):
            file_log(log_file,f"person {pid}:")
            x,y=kpts[:,0],kpts[:,1]
            for part_idx in range(0,len(parts)):
                file_log(log_file,f"part {parts(part_idx)} x:{x[part_idx]} y:{y[part_idx]}")
                if(x[part_idx]<0 or y[part_idx]<0):
                    continue
                color=colors[part_idx]
                vis_img=cv2.circle(vis_img,(int(x[part_idx]),int(y[part_idx])),radius=radius,color=color,thickness=-1)
        fig=plt.figure(figsize=(8,8))
        a=fig.add_subplot(1,2,1)
        a.set_title("original image")
        plt.imshow(ori_img)
        a=fig.add_subplot(1,2,2)
        a.set_title("visualized image")
        plt.imshow(vis_img)
        #print(f"test img_file:{type(img_file)} numpy:{type(img_file.numpy())} str:{type(str(img_file.numpy()))} ")
        image_path=bytes.decode(img_file.numpy())
        #print(f"test path:{type(image_path)} {image_path}")
        image_path=os.path.basename(image_path)
        image_mark=image_path[:image_path.rindex(".")]
        plt.savefig(f"{vis_dir}/{image_mark}_vis_{dataset_name}.png")
        plt.close('all')
        print()
    file_log(log_file,f"visualization finished! total {vis_num} image visualized!")

def basic_map_func(image_path):
    """TF Dataset pipeline."""
    # load data
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # data augmentaion using tf
    image = tf.image.random_brightness(image, max_delta=35. / 255.)  # 64./255. 32./255.)  caffe -30~50
    image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # lower=0.2, upper=1.8)  caffe 0.3~1.5
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return image
    
def get_domainadapt_targets(domainadapt_img_paths):
    domainadapt_targets=[]
    if(domainadapt_img_paths!=None):
        for _ in range(0,len(domainadapt_img_paths)):
            domainadapt_targets.append({
                "kpt":np.zeros(shape=(1,17,2)),
                "mask":None,
                "bbx":np.zeros(shape=(1,4)),
                "labeled":0
            })
    return domainadapt_targets

def get_num_parallel_calls():
    return max(multiprocessing.cpu_count()//2,1)

def log_data(msg):
    logger=logging.getLogger("DATA")
    logger.info(msg)