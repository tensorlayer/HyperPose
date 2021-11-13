import os
import cv2
import json
import numpy as np
import tensorflow as tf
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
from .processor import PostProcessor, Visualizer
from .utils import draw_results
from ..common import pad_image, resize_NCHW
from tqdm import tqdm

def multiscale_search(img,model):
    scales=[0.5,1.0,1.5,2.0]
    img_h,img_w,_=img.shape
    hin,win=model.hin,model.win
    stride=model.hin/model.hout
    data_format=model.data_format
    avg_conf_map=0
    avg_paf_map=0
    for scale in scales:
        #currently we scale image without tortion and use padding to make the image_size can be divided by stride 
        scale_rate=min(scale*model.hin/img_h,scale*model.win/img_w)
        scale_h,scale_w=int(scale_rate*img_h),int(scale_rate*img_w)
        scaled_img=cv2.resize(img,(scale_w,scale_h),interpolation=cv2.INTER_CUBIC)
        padded_img,pad=pad_image(scaled_img,stride,pad_value=0.0)
        padded_h,padded_w,_=padded_img.shape
        input_img=padded_img[np.newaxis,:,:,:].astype(np.float32)
        input_img=input_img.transpose([0,3,1,2])
        input_img=tf.convert_to_tensor(input_img)
        # image process
        conf_map,paf_map=model.infer(input_img)
        conf_map=conf_map.numpy()[0]
        paf_map=paf_map.numpy()[0]
        conf_map=np.transpose(conf_map,[1,2,0])
        paf_map=np.transpose(paf_map,[1,2,0])
        #conf_map restore
        conf_map=cv2.resize(conf_map,(padded_w,padded_h),interpolation=cv2.INTER_CUBIC)
        conf_map=conf_map[pad[0]:pad[0]+scale_h,pad[2]:pad[2]+scale_w,:]
        conf_map=cv2.resize(conf_map,(img_w,img_h),interpolation=cv2.INTER_CUBIC)
        #paf_map restore
        paf_map=cv2.resize(paf_map,(padded_w,padded_h),interpolation=cv2.INTER_CUBIC)
        paf_map=paf_map[pad[0]:pad[0]+scale_h,pad[2]:pad[2]+scale_w,:]
        paf_map=cv2.resize(paf_map,(img_w,img_h),interpolation=cv2.INTER_CUBIC)
        conf_map=np.transpose(conf_map,[2,0,1])
        paf_map=np.transpose(paf_map,[2,0,1])
        #average
        avg_conf_map+=conf_map/(len(scales))
        avg_paf_map+=paf_map/(len(scales))
    avg_conf_map = avg_conf_map[np.newaxis,:,:,:]
    avg_paf_map = avg_paf_map[np.newaxis,:,:,:]
    return avg_conf_map,avg_paf_map,input_img

def infer_one_img(model,post_processor:PostProcessor,visualizer:Visualizer,img,image_id=-1,enable_multiscale_search=False,is_visual=False):
    img=img.numpy().astype(np.float32)
    img_h,img_w,_=img.shape
    if(enable_multiscale_search):
        conf_map,paf_map,input_image=multiscale_search(img,model)
    else:
        input_image=cv2.resize(img,(model.win,model.hin))[np.newaxis,:,:,:]
        input_image=input_image.transpose([0,3,1,2])
        conf_map,paf_map=model.infer(input_image)
        conf_map=resize_NCHW(conf_map.numpy(),dst_shape=(img_h, img_w))
        paf_map=resize_NCHW(paf_map.numpy(),dst_shape=(img_h, img_w))
    predict_x = {"conf_map":conf_map, "paf_map":paf_map}
    humans = post_processor.process(predict_x, resize=False)[0]
    if(is_visual):
        visualizer.visualize(image_batch=input_image, predict_x=predict_x, name=f"{image_id}_heatmaps")
        visualizer.visualize_result(image=img, humans=humans, name=f"{image_id}_result.png")
    return humans
    
def _map_fn(image_file,image_id,hin,win):
    #load data
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image,image_id

def evaluate(model,dataset,config,vis_num=30,total_eval_num=10000,enable_multiscale_search=True):
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
    model.load_weights(os.path.join(config.model.model_dir,"newest_model.npz"), format="npz_dict")
    model.eval()
    pd_anns=[]
    vis_dir=config.eval.vis_dir
    kpt_converter=dataset.get_output_kpt_cvter()
    post_processor=PostProcessor(parts=model.parts,limbs=model.limbs,colors=model.colors,\
        hin=model.hin,win=model.win,hout=model.hout,wout=model.wout)
    visualizer = Visualizer(save_dir=f"./save_dir/{config.model.model_name}/eval_vis_dir")
    
    eval_dataset=dataset.get_eval_dataset()
    dataset_size=dataset.get_eval_datasize()
    paramed_map_fn=partial(_map_fn,hin=model.hin,win=model.win)
    eval_dataset=eval_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for eval_num,(image,image_id) in tqdm(enumerate(eval_dataset)):
        image_id=image_id.numpy()
        if(eval_num>=total_eval_num):
            break
        if(eval_num<=vis_num):
            humans=infer_one_img(model,post_processor,visualizer,image,image_id=image_id,is_visual=True,enable_multiscale_search=enable_multiscale_search)
        else:
            humans=infer_one_img(model,post_processor,visualizer,image,image_id=image_id,is_visual=False,enable_multiscale_search=enable_multiscale_search)
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(image_id)
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
            print(f"evaluating {eval_num}/{dataset_size} ...")
            
    dataset.official_eval(pd_anns,vis_dir)

def test(model,dataset,config,vis_num=30,total_test_num=10000,enable_multiscale_search=True):
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
    model.load_weights(os.path.join(config.model.model_dir,"newest_model.npz"),format="npz_dict")
    model.eval()
    pd_anns=[]
    vis_dir=config.test.vis_dir
    kpt_converter=dataset.get_output_kpt_cvter()
    post_processor=PostProcessor(parts=model.parts,limbs=model.limbs,colors=model.colors,\
        hin=model.hin,win=model.win,hout=model.hout,wout=model.wout)
    visualizer = Visualizer(save_dir=f"./save_dir/{config.model.model_name}/eval_vis_dir")
    
    test_dataset=dataset.get_test_dataset()
    dataset_size=dataset.get_test_datasize()
    paramed_map_fn=partial(_map_fn,hin=model.hin,win=model.win)
    test_dataset=test_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for test_num,(image,image_id) in enumerate(test_dataset):
        image_id=image_id.numpy()
        if(test_num>=total_test_num):
            break
        is_visual=(test_num<=vis_num)
        humans=infer_one_img(model,post_processor,image,image_id=image_id,is_visual=is_visual,enable_multiscale_search=enable_multiscale_search)
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(image_id)
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
        if(test_num%100==0):
            print(f"testing {test_num}/{dataset_size} ...")
            
    dataset.official_test(pd_anns,vis_dir)