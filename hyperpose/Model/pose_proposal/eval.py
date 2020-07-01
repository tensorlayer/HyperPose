import os
import cv2
import json
import scipy
import numpy as np
import multiprocessing
import tensorflow as tf

import matplotlib.pyplot as plt
from functools import partial
from .infer import Post_Processor
from .utils import get_parts,get_limbs,get_colors,get_output_kptcvter
from .utils import draw_bbx,draw_edge

def infer_one_img(model,post_processor,img,img_id=-1,is_visual=False,save_dir="./vis_dir/pose_proposal"):
    img=img.numpy()
    img_id=img_id.numpy()
    img_h,img_w,_=img.shape
    data_format=model.data_format
    input_img=cv2.resize(img,(model.win,model.hin))[np.newaxis,:,:,:]
    if(data_format=="channels_first"):
        input_img=np.transpose(input_img,[0,3,1,2])
    pc,pi,px,py,pw,ph,pe=model.forward(input_img,is_train=False)
    if(data_format=="channels_last"):
        pc=np.transpose(pc,[0,3,1,2])
        pi=np.transpose(pi,[0,3,1,2])
        px=np.transpose(px,[0,3,1,2])
        py=np.transpose(py,[0,3,1,2])
        pw=np.transpose(pw,[0,3,1,2])
        ph=np.transpose(ph,[0,3,1,2])
        pe=np.transpose(pe,[0,5,1,2,3,4])
    humans=post_processor.process(pc[0].numpy(),pi[0].numpy(),px[0].numpy(),py[0].numpy(),\
        pw[0].numpy(),ph[0].numpy(),pe[0].numpy())
    #resize output
    scale_w=img_w/model.win
    scale_h=img_h/model.hin
    for human in humans:
        human.scale(scale_w=scale_w,scale_h=scale_h)
    if(is_visual):
        predicts=(pc[0],px[0]*scale_w,py[0]*scale_h,pw[0]*scale_w,ph[0]*scale_h,pe[0])
        visualize(img,img_id,humans,predicts,model.hnei,model.wnei,model.hout,model.wout,post_processor.limbs,save_dir)
    return humans

def visualize(img,img_id,humans,predicts,hnei,wnei,hout,wout,limbs,save_dir):
    print(f"{len(humans)} human found!")
    print("visualizing...")
    os.makedirs(save_dir,exist_ok=True)
    img_w,img_h,_=img.shape
    pc,px,py,pw,ph,pe=predicts
    ori_img=np.clip(img*255.0,0.0,255.0).astype(np.uint8)
    #show input image
    fig=plt.figure(figsize=(8,8))
    a=fig.add_subplot(2,2,1)
    a.set_title("input image")
    plt.imshow(ori_img)
    #show output image
    vis_img=ori_img.copy()
    for human in humans:
        human.print()
        vis_img=human.draw_human(vis_img)
    a=fig.add_subplot(2,2,2)
    a.set_title("output result")
    plt.imshow(vis_img)
    #show parts and edges
    vis_img=ori_img.copy()
    vis_img=draw_bbx(vis_img,pc,px,py,pw,ph,threshold=0.7)
    vis_img=draw_edge(vis_img,pe,px,py,pw,ph,hnei,wnei,hout,wout,limbs,threshold=0.7)
    a=fig.add_subplot(2,2,3)
    a.set_title("bbxs and edges")
    plt.imshow(vis_img)
    #save result
    plt.savefig(f"{save_dir}/{img_id}_visualize.png")
    plt.close()

def _map_fn(image_file,image_id):
    #load data
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image,image_id

def evaluate(model,dataset,config,vis_num=30,total_eval_num=30):
    '''evaluate pipeline of poseProposal class models

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
    pd_anns=[]
    vis_dir=config.eval.vis_dir
    dataset_type=dataset.get_dataset_type()
    kpt_converter=get_output_kptcvter(dataset_type)
    post_processor=Post_Processor(get_parts(dataset_type),get_limbs(dataset_type),get_colors(dataset_type))
    
    eval_dataset=dataset.get_eval_dataset()
    paramed_map_fn=partial(_map_fn)
    eval_dataset=eval_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for eval_num,(img,img_id) in enumerate(eval_dataset):
        if(eval_num>=total_eval_num):
            break
        if(eval_num<=vis_num):
            humans=infer_one_img(model,post_processor,img,img_id,is_visual=True,save_dir=vis_dir)
        else:
            humans=infer_one_img(model,post_processor,img,img_id,is_visual=False,save_dir=vis_dir)
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(img_id.numpy())
            ann["id"]=human.get_global_id()
            ann["area"]=human.get_area()
            ann["score"]=human.get_score()
            kpt_list=[]
            for part_idx in range(0,len(post_processor.parts)):
                if(part_idx not in human.body_parts):
                    kpt_list.append([-1000,-1000])
                else:
                    body_part=human.body_parts[part_idx]
                    kpt_list.append([body_part.get_x(),body_part.get_y()])
            ann["keypoints"]=kpt_converter(kpt_list)
            pd_anns.append(ann)   
        #debug
        if(eval_num%10==0):
            print(f"evaluaing {eval_num}/{len(list(eval_dataset))}...")
    result_dic={"annotations":pd_anns}
    dataset.official_eval(result_dic,vis_dir)


            

        

