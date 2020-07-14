import os
import cv2
import json
import numpy as np
import tensorflow as tf
import scipy.stats as st
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
from .infer import Post_Processor
from .utils import draw_results

def infer_one_img(model,post_processor,img,img_id=-1,is_visual=False,save_dir="./vis_dir"):
    img=img.numpy()
    img_id=img_id.numpy()
    img_h,img_w,_=img.shape
    data_format=model.data_format
    input_img=cv2.resize(img.copy(),(model.win,model.hin))[np.newaxis,:,:,:]
    if(data_format=="channels_first"):
        input_img=input_img.transpose([0,3,1,2])
    conf_map,paf_map=model.forward(input_img,is_train=False)
    if(data_format=="channels_last"):
        conf_map=np.transpose(conf_map,[0,3,1,2])
        paf_map=np.transpose(paf_map,[0,3,1,2])
    conf_map=conf_map.numpy()
    paf_map=paf_map.numpy()
    humans=post_processor.process(conf_map[0],paf_map[0],img_h,img_w,data_format=data_format)
    if(is_visual): 
        draw_conf_map=cv2.resize(conf_map[0].transpose([1,2,0]),(img_w,img_h)).transpose([2,0,1])
        draw_paf_map=cv2.resize(paf_map[0].transpose([1,2,0]),(img_w,img_h)).transpose([2,0,1])
        visualize(img,img_id,humans,draw_conf_map,draw_paf_map,save_dir)
    return humans

def visualize(img,img_id,humans,conf_map,paf_map,save_dir):
    print(f"{len(humans)} human found!")
    print("visualizing...")
    os.makedirs(save_dir,exist_ok=True)
    ori_img=np.clip(img*255.0,0.0,255.0).astype(np.uint8)
    vis_img=ori_img.copy()
    for human in humans:
        vis_img=human.draw_human(vis_img)
    fig=plt.figure(figsize=(8,8))
    #show input image
    a=fig.add_subplot(2,2,1)
    a.set_title("input image")
    plt.imshow(ori_img)
    #show output result
    a=fig.add_subplot(2,2,2)
    a.set_title("output result")
    plt.imshow(vis_img)
    #show conf_map
    show_conf_map=np.amax(np.abs(conf_map[:-1,:,:]),axis=0)
    a=fig.add_subplot(2,2,3)
    a.set_title("conf_map")
    plt.imshow(show_conf_map)
    #show paf_map
    show_paf_map=np.amax(np.abs(paf_map[:,:,:]),axis=0)
    a=fig.add_subplot(2,2,4)
    a.set_title("paf_map")
    plt.imshow(show_paf_map)
    #save
    plt.savefig(f"{save_dir}/{img_id}_visualize.png")
    plt.close('all')

def _map_fn(image_file,image_id,hin,win):
    #load data
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image,image_id

def evaluate(model,dataset,config,vis_num=30,total_eval_num=30):
    '''evaluate pipeline of Openpose class models

    input model and dataset, the evaluate pipeline will start automaticly
    the evaluate pipeline will:
    1.loading newest model at path ./save_dir/model_name/model_dir/newest_model.npz
    2.perform inference and parsing over the chosen evaluate dataset
    3.visualize model output in evaluation in directory ./save_dir/model_name/eval_vis_dir
    4.output model metrics by calling dataset.official_eval()

    Parameters
    ----------
    arg1 : tensorlayer.models.MODEL
        a preset or user defined model object, obtained by Model.get_model() function
    
    arg2 : dataset
        a constructed dataset object, obtained by Dataset.get_dataset() function
    
    arg3 : Int
        an Integer indicates how many model output should be visualized
    
    arg4 : Int
        an Integer indicates how many images should be evaluated

    Returns
    -------
    None
    '''
    model.load_weights(os.path.join(config.model.model_dir,"newest_model.npz"))
    model.eval()
    pd_anns=[]
    vis_dir=config.eval.vis_dir
    kpt_converter=dataset.get_output_kpt_cvter()
    post_processor=Post_Processor(parts=model.parts,limbs=model.limbs,colors=model.colors)
    
    eval_dataset=dataset.get_eval_dataset()
    paramed_map_fn=partial(_map_fn,hin=model.hin,win=model.win)
    eval_dataset=eval_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for eval_num,(img,img_id) in enumerate(eval_dataset):
        if(eval_num>=total_eval_num):
            break
        if(eval_num<=vis_num):
            humans=infer_one_img(model,post_processor,img,img_id=img_id,is_visual=True,save_dir=vis_dir)
        else:
            humans=infer_one_img(model,post_processor,img,img_id=img_id,is_visual=False,save_dir=vis_dir)
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(img_id.numpy())
            ann["id"]=human.get_global_id()
            ann["area"]=human.get_area()
            ann["score"]=human.get_score()
            kpt_list=[]
            for part_idx in range(0,model.n_pos):
                if(part_idx not in human.body_parts):
                    kpt_list.append([-1000,-1000])
                else:
                    body_part=human.body_parts[part_idx]
                    kpt_list.append([body_part.get_x(),body_part.get_y()])
            ann["keypoints"]=kpt_converter(kpt_list)
            pd_anns.append(ann)   
        if(eval_num%100==0):
            print(f"evaluating {eval_num}/{len(list(eval_dataset))}")
            
    result_dic={"annotations":pd_anns}
    dataset.official_eval(result_dic,vis_dir)
