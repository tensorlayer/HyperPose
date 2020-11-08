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
from .utils import get_hr_conf,get_arrow_map

def infer_one_img(model,post_processor,img,image_id=-1,is_visual=False,save_dir="./save_dir"):
    img=img.numpy().astype(np.float32)
    img_id=img_id.numpy()
    #TODO: use padded-scale that wouldn't casue deformation
    img_h,img_w,_=img.shape
    input_image=cv2.resize(img,(model.win,model.hin))[np.newaxis,:,:,:]
    #default channels_first
    input_image=np.transpose(input_image,[0,3,1,2])
    pif_maps,paf_maps=model.forward(input_image,is_train=False)
    pif_maps=[pif_map[0] for pif_map in pif_maps]
    paf_maps=[paf_map[0] for paf_map in paf_maps]
    ret_humans=post_processor.process(pif_maps,paf_maps)
    if(is_visual):
        visualize(img,img_id,pif_maps,paf_maps,ret_humans,stride=post_processor.stride,save_dir=save_dir)
    return ret_humans

def visualize(img,img_id,pd_pif_maps,pd_paf_maps,humans,stride=8,save_dir=".save_dir"):
    print(f"{len(humans)} human found!")
    print("visualizing...")
    os.makedirs(save_dir,exist_ok=True)
    ori_img=np.clip(img*255.0,0.0,255.0).astype(np.uint8)
    #get ouput_result
    vis_img=ori_img.copy()
    for human in humans:
        vis_img=human.draw_human(vis_img)
    #decode result_maps
    stride=model.shape
    pd_pif_conf,pd_pif_vec,_,pd_pif_scale=pd_pif_maps
    pd_paf_conf,pd_paf_src_vec,pd_paf_dst_vec,_,_,_,_=pd_paf_maps
    pd_pif_conf_show=np.amax(pd_pif_conf,axis=0)
    pd_pif_hr_conf_show=np.amax(get_hr_conf(pd_pif_conf,pd_pif_vec,pd_pif_scale,stride=stride,thresh=0.1),axis=0)
    pd_paf_conf_show=np.amax(pd_paf_conf,axis=0)
    pd_paf_vec_show=np.zeros(pd_pif_hr_conf_show.shape[0],pd_pif_hr_conf_show.shape[1],3)
    pd_paf_vec_show=get_arrow_map(pd_paf_vec_show,pd_paf_conf,pd_paf_src_vec,pd_paf_dst_vec,thresh=0.1)
    #plt draw
    fig=plt.figure(figsize=(12,12))
    #show input image
    a=fig.add_subplot(2,3,1)
    a.set_title("input image")
    plt.imshow(ori_img)
    #show output result
    a=fig.add_subplot(2,3,4)
    a.set_title("output result")
    plt.imshow(vis_img)
    #show pif_conf_map
    a=fig.add_subplot(2,3,2)
    a.set_title("pif_conf_map")
    plt.imshow(pd_pif_conf_show,alpha=0.8)
    plt.colorbar()
    #show pif_hr_conf_map
    a=fig.add_subplot(2,3,3)
    a.set_title("pif_hr_conf_map")
    plt.imshow(pd_pif_hr_conf_show,alpha=0.8)
    plt.colorbar()
    #show paf_conf_map
    a=fig.add_subplot(2,3,5)
    a.set_title("paf_conf_map")
    plt.imshow(pd_paf_conf_show,alpha=0.8)
    plt.colorbar()
    #show paf_vec_map
    a=fig.add_subplot(2,3,6)
    a.set_title("paf_vec_map")
    plt.imshow(pd_paf_vec_show,alpha=0.8)
    plt.colorbar()
    #save fig
    plt.savefig(os.path.join(save_dir,f"{image_id}_visualize.png"))
    plt.close()

def _map_fn(image_file,image_id,hin,win):
    #load data
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image,image_id

def evaluate(model,dataset,config,vis_num=30,total_eval_num=10000,enable_multiscale_search=False):
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
    print(f"enable multiscale_search:{enable_multiscale_search}")
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
            humans=infer_one_img(model,post_processor,img,img_id=img_id,is_visual=True,save_dir=vis_dir,enable_multiscale_search=enable_multiscale_search)
        else:
            humans=infer_one_img(model,post_processor,img,img_id=img_id,is_visual=False,save_dir=vis_dir,enable_multiscale_search=enable_multiscale_search)
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