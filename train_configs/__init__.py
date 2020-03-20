import os
import logging
import tensorflow as tf
import tensorlayer as tl
from .train_config_lopps import config as config_lopps
from .train_config_ppn import config as config_ppn

def init_config(model_type,model_name="default_name",dataset_path="data"):
    #get basic config
    if(model_type=="lightweight_openpose"):
        config=config_lopps
    elif(model_type=="pose_proposal"):
        config=config_ppn
    else:
        raise RuntimeError(f'unknown model type {model_type}')

    #model configure
    config.MODEL.model_type=model_type 
    config.MODEL.model_name=model_name
    print(f"model_type:{config.MODEL.model_type}")
    print(f"model_name:{config.MODEL.model_name}")
    config.MODEL.model_dir=f"save_dir/{model_name}/model_dir"
    tl.files.exists_or_mkdir(config.MODEL.model_dir, verbose=True)  # to save model files

    #dataset configure
    config.DATA.data_path=dataset_path

    #vis_dir configure
    config.LOG.vis_dir=f"save_dir/{model_name}/vis_dir"
    tl.files.exists_or_mkdir(config.LOG.vis_dir, verbose=True)  # to save visualization results

    #device configure
    #FIXME: replace experimental tf functions when in tf 2.1 version
    tf.debugging.set_log_device_placement(False)
    tf.config.set_soft_device_placement(True)
    for gpu in tf.config.experimental.get_visible_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu,True)
        
    #logging configure
    config.LOG.log_path=f"save_dir/{model_name}/log.txt"
    tl.files.exists_or_mkdir(os.path.dirname(config.LOG.log_path),verbose=True)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    tl.logging.set_verbosity(tl.logging.INFO)
    
    print(f"configure initialization finished!")
    return config
