import os
import cv2
import numpy as np
import multiprocessing
import tensorflow as tf

import matplotlib.pyplot as plt
from functools import partial
from .processor import PostProcessor
from .utils import draw_bbx,draw_edge
from tqdm import tqdm

def infer_one_img(model,postprocessor,img,img_id=-1,is_visual=False,save_dir="./vis_dir/pose_proposal"):
    img=img.numpy()
    img_h,img_w,img_c=img.shape
    data_format=model.data_format
    scale_rate=min(model.hin/img_h,model.win/img_w)
    scale_w,scale_h=int(img_w*scale_rate),int(img_h*scale_rate)
    resize_img=cv2.resize(img,(scale_w,scale_h))
    input_img=np.zeros(shape=(model.win,model.hin,img_c))
    input_img[0:scale_h,0:scale_w,:]=resize_img
    input_img=input_img[np.newaxis,:,:,:].astype(np.float32)
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
    humans=postprocessor.process(pc[0].numpy(),pi[0].numpy(),px[0].numpy(),py[0].numpy(),\
        pw[0].numpy(),ph[0].numpy(),pe[0].numpy(),scale_w_rate=scale_rate,scale_h_rate=scale_rate)
    #resize output
    for human in humans:
        human.scale(scale_w=1/scale_rate,scale_h=1/scale_rate)
    if(is_visual):
        predicts=(pc[0],px[0]/scale_rate,py[0]/scale_rate,pw[0]/scale_rate,ph[0]/scale_rate,pe[0])
        visualize(img,img_id,humans,predicts,model.hnei,model.wnei,model.hout,model.wout,postprocessor.limbs,save_dir)
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
    vis_img=draw_bbx(vis_img,pc,px,py,pw,ph,threshold=0.3)
    vis_img=draw_edge(vis_img,pe,px,py,pw,ph,hnei,wnei,hout,wout,limbs,threshold=0.3)
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

def evaluate(model,dataset,config,vis_num=30,total_eval_num=10000,enable_multiscale_search=False):
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
    model.load_weights(os.path.join(config.model.model_dir,"newest_model.npz"), format="npz_dict")
    pd_anns=[]
    vis_dir=config.eval.vis_dir
    kpt_converter=dataset.get_output_kpt_cvter()
    postprocessor=PostProcessor(model.parts,model.limbs,model.colors)
    
    eval_dataset=dataset.get_eval_dataset()
    dataset_size=dataset.get_eval_datasize()
    paramed_map_fn=partial(_map_fn)
    eval_dataset=eval_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for eval_num,(img,img_id) in tqdm(enumerate(eval_dataset)):
        img_id=img_id.numpy()
        if(eval_num>=total_eval_num):
            break
        if(eval_num<=vis_num):
            humans=infer_one_img(model,postprocessor,img,img_id,is_visual=True,save_dir=vis_dir)
        else:
            humans=infer_one_img(model,postprocessor,img,img_id,is_visual=False,save_dir=vis_dir)
        if(len(humans)==0):
            pd_anns.append({"category_id":1,"image_id":int(img_id),"id":-1,\
            "area":-1,"score":-1,"keypoints":[0,0,-1]*len(dataset.get_parts())})
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(img_id)
            ann["id"]=human.get_global_id()
            ann["area"]=human.get_area()
            ann["score"]=human.get_score()
            kpt_list=[]
            for part_idx in range(0,len(postprocessor.parts)):
                if(part_idx not in human.body_parts):
                    kpt_list.append([-1000,-1000])
                else:
                    body_part=human.body_parts[part_idx]
                    kpt_list.append([body_part.get_x(),body_part.get_y()])
            ann["keypoints"]=kpt_converter(kpt_list)
            pd_anns.append(ann)   
        #debug
        if(eval_num%10==0):
            print(f"evaluaing {eval_num}/{dataset_size} ...")

    dataset.official_eval(pd_anns,vis_dir)

def test(model,dataset,config,vis_num=30,total_test_num=10000,enable_multiscale_search=False):
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
    vis_dir=config.test.vis_dir
    kpt_converter=dataset.get_output_kpt_cvter()
    postprocessor=PostProcessor(model.parts,model.limbs,model.colors)
    
    test_dataset=dataset.get_test_dataset()
    dataset_size=dataset.get_test_datasize()
    paramed_map_fn=partial(_map_fn)
    test_dataset=test_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for test_num,(img,img_id) in enumerate(test_dataset):
        img_id=img_id.numpy()
        if(test_num>=total_test_num):
            break
        is_visual=(test_num<=vis_num)
        humans=infer_one_img(model,postprocessor,img,img_id,is_visual=is_visual,save_dir=vis_dir)
        if(len(humans)==0):
            pd_anns.append({"category_id":1,"image_id":int(img_id),"id":-1,\
            "area":-1,"score":-1,"keypoints":[0,0,-1]*len(dataset.get_parts())})
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(img_id)
            ann["id"]=human.get_global_id()
            ann["area"]=human.get_area()
            ann["score"]=human.get_score()
            kpt_list=[]
            for part_idx in range(0,len(postprocessor.parts)):
                if(part_idx not in human.body_parts):
                    kpt_list.append([-1000,-1000])
                else:
                    body_part=human.body_parts[part_idx]
                    kpt_list.append([body_part.get_x(),body_part.get_y()])
            ann["keypoints"]=kpt_converter(kpt_list)
            pd_anns.append(ann)   
        #debug
        if(test_num%10==0):
            print(f"evaluaing {test_num}/{dataset_size} ...")

    dataset.official_test(pd_anns,vis_dir)