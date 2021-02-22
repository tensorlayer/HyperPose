import os
import logging
import matplotlib
matplotlib.use("Agg")
from copy import deepcopy
from easydict import EasyDict as edict
from .define import *
from .config_pretrain import pretrain
update_config,update_train,update_eval,update_test,update_model,update_data,update_log=edict(),edict(),edict(),edict(),edict(),edict(),edict()

#default train
update_train.optim_type=OPTIM.Adam

#defualt model config
update_model.model_type=MODEL.Openpose
#userdef model
update_model.userdef_parts=None
update_model.userdef_limbs=None

#default dataset config
#official dataset
update_data.official_flag=True
#userdef dataset
update_data.userdef_dataset=None
#useradd dataset
update_data.useradd_flag=False
update_data.useradd_scale_rate=1
update_data.useradd_train_img_paths=None
update_data.useradd_train_targets=None
#domain adaption dataset
update_data.domainadapt_flag=False
update_data.domainadapt_scale_rate=1
update_data.domainadapt_train_img_paths=None
#default pretrain config
update_pretrain=edict()


#get configure api
def get_config():
    '''get the config object with all the configuration information

    get the config object based on the previous setting functions, 
    the config object will be passed to the functions of Model and Dataset module to
    construct the system.

    only the setting functions called before this get_config function is valid, thus
    use this function after all configuration done.
    
    Parameters
    ----------
    None

    Returns
    -------
    config object
        an edict object contains all the configuration information.

    '''
    #import basic configurations
    if(update_model.model_type==MODEL.Openpose):
        from .config_opps import model,train,eval,test,data,log
    elif(update_model.model_type==MODEL.LightweightOpenpose):
        from .config_lopps import model,train,eval,test,data,log
    elif(update_model.model_type==MODEL.MobilenetThinOpenpose):
        from .config_mbtopps import model,train,eval,test,data,log
    elif(update_model.model_type==MODEL.PoseProposal):
        from .config_ppn import model,train,eval,test,data,log
    elif(update_model.model_type==MODEL.Pifpaf):
        from .config_pifpaf import model,train,eval,test,data,log
    #merge settings with basic configurations
    model.update(update_model)
    train.update(update_train)
    eval.update(update_eval)
    test.update(update_test)
    data.update(update_data)
    log.update(update_log)
    pretrain.update(update_pretrain)
    #assemble configure
    config=edict()
    config.model=model
    config.train=train
    config.eval=eval
    config.test=test
    config.data=data
    config.log=log
    config.pretrain=pretrain
    #path configure
    import tensorflow as tf
    import tensorlayer as tl
    tl.files.exists_or_mkdir(config.model.model_dir, verbose=True)  # to save model files 
    tl.files.exists_or_mkdir(config.train.vis_dir, verbose=True)  # to save visualization results
    tl.files.exists_or_mkdir(config.eval.vis_dir, verbose=True)  # to save visualization results
    tl.files.exists_or_mkdir(config.test.vis_dir, verbose=True)  # to save visualization results
    tl.files.exists_or_mkdir(config.data.vis_dir, verbose=True)  # to save visualization results
    tl.files.exists_or_mkdir(config.pretrain.pretrain_model_dir,verbose=True)
    #device configure
    #FIXME: replace experimental tf functions when in tf 2.1 version
    tf.debugging.set_log_device_placement(False)
    tf.config.set_soft_device_placement(True)
    for gpu in tf.config.experimental.get_visible_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu,True)
    #limit the cpu usage when pretrain
    #logging configure
    tl.files.exists_or_mkdir(os.path.dirname(config.log.log_path),verbose=True)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)
    tl.logging.set_verbosity(tl.logging.WARN)
    return deepcopy(config)

#set configure api
#model configure api
def set_model_arch(model_arch):
    '''set user defined model architecture

    replace default model architecture with user-defined model architecture, use it in the following training and evaluation 
    
    Parameters
    ----------
    arg1 : tensorlayer.models.MODEL
        An object of a model class inherit from tensorlayer.models.MODEL class,
        should implement forward function and cal_loss function to make it compatible with the existing pipeline

        The forward funtion should follow the signature below:
|           openpose models: def forward(self,x,is_train=False) ,return conf_map,paf_map,stage_confs,stage_pafs
|           poseproposal models: def forward(self,x,is_train=False), return pc,pi,px,py,pw,ph,pe

        The cal_loss function should follow the signature below:

|           openpose models: def cal_loss(self,stage_confs,stage_pafs,gt_conf,gt_paf,mask), return loss,loss_confs,loss_pafs
|           poseproposal models: def cal_loss(self,tc,tx,ty,tw,th,te,te_mask,pc,pi,px,py,pw,ph,pe):
         return loss_rsp,loss_iou,loss_coor,loss_size,loss_limb
    
    Returns
    -------
    None

    '''
    update_model.model_arch=model_arch

def set_model_type(model_type):
    '''set preset model architecture 
    
    configure the model architecture as one of the desired preset model architectures

    Parameters
    ----------
    arg1 : Config.MODEL
        a enum value of enum class Config.MODEL, available options:
|            Config.MODEL.Openpose (original Openpose)
|            Config.MODEL.LightweightOpenpose (lightweight variant version of Openpose,real-time on cpu)
|            Config.MODEL.PoseProposal (pose proposal network)
|            Config.MODEL.MobilenetThinOpenpose (lightweight variant version of openpose)
    
    Returns
    -------
    None
    '''
    update_model.model_type=model_type


def set_model_backbone(model_backbone):
    '''set preset model backbones 
    
    set current model backbone to other common backbones 
    different backbones have different computation complexity
    this enable dynamicly adapt the model architecture to approriate size.

    Parameters
    ----------
    arg1 : Config.BACKBONE
        a enum value of enum class Config.BACKBONE
        available options:
|           Config.BACKBONE.DEFUALT (default backbone of the architecture)
|           Config.BACKBONE.MobilenetV1
|           Config.BACKBONE.MobilenetV2
|           Config.BACKBONE.Vggtiny
|           Config.BACKBONE.Vgg16
|           Config.BACKBONE.Vgg19
|           Config.BACKBONE.Resnet18
|           Config.BACKBONE.Resnet50
    
    Returns
    -------
    None
    '''
    update_model.model_backbone=model_backbone

def set_data_format(data_format):
    '''set model dataformat

    set the channel order of current model:

|       "channels_first" dataformat is faster in deployment
|       "channels_last" dataformat is more common
    the integrated pipeline will automaticly adapt to the chosen data format

    Parameters
    ----------
    arg1 : string
        available input:
|           'channels_first': data_shape N*C*H*W
|           'channels_last': data_shape N*H*W*C  
    
    Returns
    -------
    None
    '''
    update_model.data_format=data_format

def set_model_name(model_name):
    '''set the name of model

    the models are distinguished by their names,so it is necessary to set model's name when train multiple models at the same time.
    each model's ckpt data and log are saved on the 'save_dir/model_name' directory, the following directory are determined:

|       directory to save model                      ./save_dir/model_name/model_dir
|       directory to save train result               ./save_dir/model_name/train_vis_dir
|       directory to save evaluate result            ./save_dir/model_name/eval_vis_dir
|       directory to save dataset visualize result   ./save_dir/model_name/data_vis_dir
|       file path to save train log                  ./save_dir/model_name/log.txt

    Parameters
    ----------
    arg1 : string
        name of the model
    
    Returns
    -------
    None
    '''
    update_model.model_name=model_name
    update_model.model_dir = f"./save_dir/{update_model.model_name}/model_dir"
    update_train.vis_dir = f"./save_dir/{update_model.model_name}/train_vis_dir"
    update_eval.vis_dir=f"./save_dir/{update_model.model_name}/eval_vis_dir"
    update_test.vis_dir=f"./save_dir/{update_model.model_name}/test_vis_dir"
    update_data.vis_dir=f"./save_dir/{update_model.model_name}/data_vis_dir"
    update_log.log_path= f"./save_dir/{update_model.model_name}/log.txt"

def set_model_parts(userdef_parts):
    update_model.userdef_parts=userdef_parts

def set_model_limbs(userdef_limbs):
    update_model.userdef_limbs=userdef_limbs

#train configure api
def set_train_type(train_type):
    '''set single_train or parallel train

    default using single train, which train the model on one GPU.
    set parallel train will use Kungfu library to accelerate training on multiple GPU.

    to use parallel train better, it is also allow to set parallel training optimizor by set_kungfu_option.

    Parameters
    ----------
    arg1 : Config.TRAIN
        a enum value of enum class Config.TRAIN,available options:
|           Config.TRAIN.Single_train
|           Config.TRAIN.Parallel_train
    
    Returns
    -------
    None
    '''
    update_train.train_type=train_type

def set_optim_type(optim_type):
    update_train.optim_type=optim_type

def set_learning_rate(learning_rate):
    '''set the learning rate in training

    Parameters
    ----------
    arg1 : float
        learning rate
    
    Returns
    -------
    None
    '''
    update_train.lr_init=learning_rate

def set_save_interval(save_interval):
    update_train.save_interval=save_interval

def set_batch_size(batch_size):
    '''set the batch size in training

    Parameters
    ----------
    arg1 : int
        batch_size
    
    Returns
    -------
    None
    '''
    update_train.batch_size=batch_size


def set_kungfu_option(kungfu_option):
    '''set the optimizor of parallel training

    kungfu distribute training library needs to wrap tensorflow optimizor in
    kungfu optimizor, this function is to choose kungfu optimizor wrap type

    Parameters
    ----------
    arg1 : Config.KUNGFU
        a enum value of enum class Config.KUNGFU
        available options:
|           Config.KUNGFU.Sync_sgd (SynchronousSGDOptimizer, hyper-parameter-robus)
|           Config.KUNGFU.Sync_avg (SynchronousAveragingOptimizer)
|           Config.KUNGFU.Pair_avg (PairAveragingOptimizer, communication-efficient)
    
    Returns
    -------
    None
    '''
    update_train.kungfu_option=kungfu_option

#data configure api
def set_dataset_type(dataset_type):
    '''set the dataset for train and evaluate

    set which dataset to use, the process of downlaoding, decoding, reformatting of different type
    of dataset is automatic.
    the evaluation metric of different dataset follows their official metric,
    for COCO is MAP, for MPII is MPCH.
    
    This API also receive user-defined dataset class, which should implement the following functions
|       __init__: take the config object with all configuration to init the dataset
|       get_parts: return a enum class which defines the key point definition of the dataset
|       get_limbs: return a [2*num_limbs] array which defines the limb definition of the dataset
|       get_colors: return a list which defines the visualization color of the limbs
|       get_train_dataset: return a tensorflow dataset which contains elements for training. each element should contains an image path and a target dict decoded in bytes by _pickle
|       get_eval_dataset: return a tensorflow dataset which contains elements for evaluating. each element should contains an image path and an image id
|       official_eval: if want to evaluate on this user-defined dataset, evalutation function should be implemented.
    one can refer the Dataset.mpii_dataset and Dataset.mscoco_dataset for detailed information.

    Parameters
    ----------
    arg1 : Config.DATA
        a enum value of enum class Config.DATA or user-defined dataset
        available options:
|           Config.DATA.MSCOCO
|           Config.DATA.MPII
|           user-defined dataset
    
    Returns
    -------
    None
    '''
    update_data.dataset_type=dataset_type

def set_dataset_version(dataset_version):
    update_data.dataset_version=dataset_version

def set_useradd_data(useradd_train_img_paths,useradd_train_targets,useradd_scale_rate=1):
    update_data.useradd_flag=True
    update_data.useradd_train_img_paths=useradd_train_img_paths
    update_data.useradd_train_targets=useradd_train_targets
    update_data.useradd_scale_rate=useradd_scale_rate

def set_userdef_dataset(userdef_dataset):
    update_data.dataset_type=DATA.USERDEF
    update_data.userdef_dataset=userdef_dataset

def set_multiple_dataset(multiple_dataset_configs):
    update_data.dataset_type=DATA.MULTIPLE
    update_data.multiple_dataset_configs=multiple_dataset_configs

def set_domainadapt_dataset(domainadapt_train_img_paths,domainadapt_scale_rate=1):
    update_data.domainadapt_flag=True
    update_data.domainadapt_train_img_paths=domainadapt_train_img_paths
    update_data.domainadapt_scale_rate=domainadapt_scale_rate

def set_official_dataset(official_flag):
    update_data.official_flag=official_flag

def set_dataset_path(dataset_path):
    '''set the path of the dataset

    set the path of the directory where dataset is,if the dataset doesn't exist in this directory, 
    then it will be automaticly download in this directory and decoded. 

    Parameters
    ----------
    arg1 : String
        a string indicates the path of the dataset,
        default: ./data 
    
    Returns
    -------
    None
    '''
    update_data.dataset_path=dataset_path

def set_dataset_filter(dataset_filter):
    '''set the user defined dataset filter

    set the dataset filter as the input function.
    to uniformly format different dataset, 
    Hyperpose organize the annotations of one image in one dataset in the similiar meta classes.
    for COCO dataset, it is COCOMeta; for MPII dataset, it is MPIIMeta.
    Meta classes will have some common information such as image_id, joint_list etc,
    they also have some dataset-specific imformation, such as mask, is_crowd, headbbx_list etc.
    
    the dataset_fiter will perform on the Meta objects of the corresponding dataset, if 
    it returns True, the image and annotaions the Meta object related will be kept,
    otherwise it will be filtered out. Please refer the Dataset.xxxMeta classes for better use.

    Parameters
    ----------
    arg1 : function
        a function receive a meta object as input, return a bool value indicates whether 
        the meta should be kept or filtered out. return Ture for keeping and False for depricating the object.
        default: None
    
    Returns
    -------
    None
    '''
    update_data.dataset_filter=dataset_filter

#log configure api
def set_log_interval(log_interval):
    '''set the frequency of logging

    set the how many iteration intervals between two log information

    Parameters
    ----------
    arg1 : Int
        a int value indicates the iteration number bwteen two logs
        default: 1 
    
    Returns
    -------
    None
    '''
    update_log.log_interval=log_interval

def set_pretrain(enable):
    update_pretrain.enable=enable

def set_pretrain_dataset_path(pretrain_dataset_path):
    update_pretrain.pretrain_dataset_path=pretrain_dataset_path
