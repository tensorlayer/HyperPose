import os
import cv2
import numpy as np
import tensorflow as tf
from functools import partial
import multiprocessing
import matplotlib.pyplot as plt
from .processor import PostProcessor, Visualizer
from .utils import get_hr_conf,get_arrow_map,maps_to_numpy
from ..common import pad_image_shape

def infer_one_img(model,postprocessor:PostProcessor,visualizer:Visualizer,img,img_id=-1,is_visual=False,enable_multiscale_search=False,debug=False):
    img=img.numpy().astype(np.float32)
    if(debug):
        print(f"infer image id:{img_id}")
        print(f"infer image shape:{img.shape}")
    #TODO: use padded-scale that wouldn't casue deformation
    img_h,img_w,_=img.shape
    hin,win=model.hin,model.win
    scale_rate=min(hin/img_h,win/img_w)*0.95
    scale_h,scale_w=int(scale_rate*img_h),int(scale_rate*img_w)
    scale_image=cv2.resize(img,(scale_w,scale_h),interpolation=cv2.INTER_CUBIC)
    padded_image,pad=pad_image_shape(scale_image,shape=(hin,win),pad_value=0.0)
    #default channels_first
    input_image=np.transpose(padded_image[np.newaxis,:,:,:].astype(np.float32),[0,3,1,2])
    predict_x = model.forward(input_image,is_train=False)
    humans=postprocessor.process(predict_x)[0]
    for human in humans:
        human.bias(bias_w=-pad[2],bias_h=-pad[0])
        human.scale(scale_w=1/scale_rate,scale_h=1/scale_rate)
    if(is_visual):
        visualizer.visualize(image_batch=input_image, predict_x=predict_x, name=f"{img_id}_heatmap")
        visualizer.visualize_result(image=img, humans=humans, name=f"{img_id}_result")
    return humans

def visualize(img,img_id,processed_img,pd_pif_maps,pd_paf_maps,humans,stride=8,save_dir="./save_dir"):
    print(f"{len(humans)} human found!")
    print("visualizing...")
    os.makedirs(save_dir,exist_ok=True)
    ori_img=np.clip(img*255.0,0.0,255.0).astype(np.uint8)
    processed_img=np.clip(processed_img*255.0,0.0,255.0).astype(np.uint8)
    #get ouput_result
    vis_img=ori_img.copy()
    for human in humans:
        vis_img=human.draw_human(vis_img)
    #decode result_maps
    pd_pif_conf,pd_pif_vec,_,pd_pif_scale=pd_pif_maps
    pd_paf_conf,pd_paf_src_vec,pd_paf_dst_vec,_,_,_,_,=pd_paf_maps
    pd_pif_conf_show=np.amax(pd_pif_conf,axis=0)
    pd_pif_hr_conf_show=np.amax(get_hr_conf(pd_pif_conf,pd_pif_vec,pd_pif_scale,stride=stride,thresh=0.1),axis=0)
    pd_paf_conf_show=np.amax(pd_paf_conf,axis=0)
    pd_paf_vec_show=np.zeros(shape=(pd_pif_hr_conf_show.shape[0],pd_pif_hr_conf_show.shape[1],3)).astype(np.int8)
    pd_paf_vec_show=get_arrow_map(pd_paf_vec_show,pd_paf_conf,pd_paf_src_vec,pd_paf_dst_vec,thresh=0.1)
    #plt draw
    fig=plt.figure(figsize=(12,12))
    #show input image
    a=fig.add_subplot(3,3,1)
    a.set_title("input image")
    plt.imshow(ori_img)
    #show output result
    a=fig.add_subplot(3,3,3)
    a.set_title("output result")
    plt.imshow(vis_img)
    #pif
    #show processed image
    a=fig.add_subplot(3,3,4)
    a.set_title("processed image")
    plt.imshow(processed_img)
    #show pif_conf_map
    a=fig.add_subplot(3,3,5)
    a.set_title("pif_conf_map")
    plt.imshow(pd_pif_conf_show,alpha=0.8)
    plt.colorbar()
    #show pif_hr_conf_map
    a=fig.add_subplot(3,3,6)
    a.set_title("pif_hr_conf_map")
    plt.imshow(pd_pif_hr_conf_show,alpha=0.8)
    plt.colorbar()
    #paf
    a=fig.add_subplot(3,3,7)
    a.set_title("processed image")
    plt.imshow(processed_img)
    #show paf_conf_map
    a=fig.add_subplot(3,3,8)
    a.set_title("paf_conf_map")
    plt.imshow(pd_paf_conf_show,alpha=0.8)
    plt.colorbar()
    #show paf_vec_map
    a=fig.add_subplot(3,3,9)
    a.set_title("paf_vec_map")
    plt.imshow(pd_paf_vec_show,alpha=0.8)
    plt.colorbar()
    #save fig
    plt.savefig(os.path.join(save_dir,f"{img_id}_visualize.png"))
    plt.close()

def _map_fn(image_file,image_id):
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
    model.load_weights(os.path.join(config.model.model_dir,"newest_model.npz"),format="npz_dict")
    model.eval()
    pd_anns=[]
    vis_dir=config.eval.vis_dir
    kpt_converter=dataset.get_output_kpt_cvter()
    postprocessor=PostProcessor(parts=model.parts,limbs=model.limbs,colors=model.colors,hin=model.hin,win=model.win,\
        hout=model.hout,wout=model.wout,debug=False)
    visualizer = Visualizer(save_dir=vis_dir)
    
    eval_dataset=dataset.get_eval_dataset()
    dataset_size=dataset.get_eval_datasize()
    paramed_map_fn=_map_fn
    eval_dataset=eval_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for eval_num,(img,img_id) in enumerate(eval_dataset):
        img_id=img_id.numpy()
        if(eval_num>=total_eval_num):
            break
        is_visual=(eval_num<=vis_num)
        humans=infer_one_img(model,postprocessor,visualizer,img,img_id=img_id,is_visual=is_visual,enable_multiscale_search=enable_multiscale_search)
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(img_id)
            ann["id"]=human.get_global_id()
            ann["area"]=human.get_area()
            ann["score"]=human.get_score()
            kpt_list=[]
            for part_idx in range(0,model.n_pos):
                if(part_idx not in human.body_parts):
                    kpt_list.append([-1000.0,-1000.0])
                else:
                    body_part=human.body_parts[part_idx]
                    kpt_list.append([body_part.get_x(),body_part.get_y()])
            ann["keypoints"]=kpt_converter(kpt_list)
            pd_anns.append(ann)   
        if(eval_num%100==0):
            print(f"evaluating {eval_num}/{dataset_size} ...")
    
    dataset.official_eval(pd_anns,vis_dir)

def test(model,dataset,config,vis_num=30,total_test_num=10000,enable_multiscale_search=False):
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
    postprocessor=PostProcessor(parts=model.parts,limbs=model.limbs,colors=model.colors,hin=model.hin,win=model.win,\
        hout=model.hout,wout=model.wout,debug=False)
    visualizer = Visualizer(save_dir=vis_dir)
    
    test_dataset=dataset.get_test_dataset()
    dataset_size=dataset.get_test_datasize()
    paramed_map_fn=_map_fn
    test_dataset=test_dataset.map(paramed_map_fn,num_parallel_calls=max(multiprocessing.cpu_count()//2,1))
    for test_num,(img,img_id) in enumerate(test_dataset):
        img_id=img_id.numpy()
        if(test_num>=total_test_num):
            break
        is_visual=(test_num<=vis_num)
        humans=infer_one_img(model,postprocessor,visualizer,img,img_id=img_id,is_visual=is_visual,enable_multiscale_search=enable_multiscale_search)
        for human in humans:
            ann={}
            ann["category_id"]=1
            ann["image_id"]=int(img_id)
            ann["id"]=human.get_global_id()
            ann["area"]=human.get_area()
            ann["score"]=human.get_score()
            kpt_list=[]
            for part_idx in range(0,model.n_pos):
                if(part_idx not in human.body_parts):
                    kpt_list.append([-1000.0,-1000.0])
                else:
                    body_part=human.body_parts[part_idx]
                    kpt_list.append([body_part.get_x(),body_part.get_y()])
            ann["keypoints"]=kpt_converter(kpt_list)
            pd_anns.append(ann)   
        if(test_num%100==0):
            print(f"testing {test_num}/{dataset_size} ...")
    
    dataset.official_test(pd_anns,vis_dir)
