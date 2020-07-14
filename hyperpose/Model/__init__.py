import tensorflow as tf
from functools import partial
from .human import Human
from .common import rename_tensor
from .common import TRAIN,MODEL,DATA,KUNGFU,BACKBONE
from .openpose import OpenPose,LightWeightOpenPose,MobilenetThinOpenpose
from .pose_proposal import PoseProposal
from .backbones import MobilenetV1_backbone,MobilenetV2_backbone,vgg16_backbone,vgg19_backbone,vggtiny_backbone
from .backbones import Resnet18_backbone,Resnet50_backbone

def get_model(config):
    '''get model based on config object

    construct and return a model based on the configured model_type and model_backbone.
    each preset model architecture has a default backbone, replace it with chosen common model_backbones allow user to
    change model computation complexity to adapt to application scene.

    Parameters
    ----------
    arg1 : config object
        the config object return by Config.get_config() function, which includes all the configuration information.
    
    Returns
    -------
    tensorlayer.models.MODEL
        a model object inherited from tensorlayer.models.MODEL class, has configured model architecture and chosen 
        model backbone. can be user defined architecture by using Config.set_model_architecture() function.
    '''
    model=config.model
    #user configure model arch themselves
    if("model_arch" in model):
        print("using user self-defined model arch!")
        ret_model=model.model_arch(config)
    #using defualt arch
    else:
        backbone=None
        if("model_backbone" in model):
            model_backbone=model.model_backbone
            if(model_backbone==BACKBONE.Default):
                print(f"using default backbone!")
            elif(model_backbone==BACKBONE.Mobilenetv1):
                backbone=MobilenetV1_backbone
                print(f"setting MobilnetV1_backbone!")
            elif(model_backbone==BACKBONE.Vgg19):
                backbone=vgg19_backbone
                print(f"setting Vgg19_backbone!")
            elif(model_backbone==BACKBONE.Resnet18):
                backbone=Resnet18_backbone
                print(f"setting Resnet18_backbone!")
            elif(model_backbone==BACKBONE.Resnet50):
                backbone=Resnet50_backbone
                print(f"setting Resnet50_backbone!")
            elif(model_backbone==BACKBONE.Vggtiny):
                backbone=vggtiny_backbone
                print(f"setting Vggtiny_backbone")
            elif(model_backbone==BACKBONE.Mobilenetv2):
                backbone=MobilenetV2_backbone
                print(f"setting MobilenetV2_backbone")
            elif(model_backbone==BACKBONE.Vgg16):
                backbone=vgg16_backbone
                print(f"setting Vgg16_backbone")
            else:
                raise NotImplementedError(f"unknown model backbone {model_backbone}")

        model_type=model.model_type
        dataset_type=config.data.dataset_type
        if(model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose):
            from .openpose.utils import get_parts
            from .openpose.utils import get_limbs
            model.parts=get_parts(dataset_type)
            model.limbs=get_limbs(dataset_type)
        elif(model_type == MODEL.PoseProposal):
            from .pose_proposal.utils import get_parts
            from .pose_proposal.utils import get_limbs
            model.parts=get_parts(dataset_type)
            model.limbs=get_limbs(dataset_type)
        
        #set model
        if model_type == MODEL.Openpose:
            from .openpose import OpenPose as model_arch
            ret_model=model_arch(parts=model.parts,n_pos=len(model.parts),limbs=model.limbs,n_limbs=len(model.limbs),num_channels=model.num_channels,\
                hin=model.hin,win=model.win,hout=model.hout,wout=model.wout,backbone=backbone,data_format=model.data_format)
        elif model_type == MODEL.LightweightOpenpose:
            from .openpose import LightWeightOpenPose as model_arch
            ret_model=model_arch(parts=model.parts,n_pos=len(model.parts),limbs=model.limbs,n_limbs=len(model.limbs),num_channels=model.num_channels,hin=model.hin,win=model.win,\
                hout=model.hout,wout=model.wout,backbone=backbone,data_format=model.data_format)
        elif model_type == MODEL.MobilenetThinOpenpose:
            from .openpose import MobilenetThinOpenpose as model_arch
            ret_model=model_arch(parts=model.parts,n_pos=len(model.parts),limbs=model.limbs,n_limbs=len(model.limbs),num_channels=model.num_channels,hin=model.hin,win=model.win,\
                hout=model.hout,wout=model.wout,backbone=backbone,data_format=model.data_format)
        elif model_type == MODEL.PoseProposal:
            from .pose_proposal import PoseProposal as model_arch
            ret_model=model_arch(parts=model.parts,K_size=len(model.parts),limbs=model.limbs,L_size=len(model.limbs),hnei=model.hnei,wnei=model.wnei,lmd_rsp=model.lmd_rsp,\
                lmd_iou=model.lmd_iou,lmd_coor=model.lmd_coor,lmd_size=model.lmd_size,lmd_limb=model.lmd_limb,backbone=backbone,\
                data_format=model.data_format)
        else:
            raise RuntimeError(f'unknown model type {model_type}')
        print(f"using {model_type.name} model arch!")
    return ret_model

def get_train(config):
    '''get train pipeline based on config object

    construct train pipeline based on the chosen model_type and dataset_type,
    default is single train pipeline performed on single GPU,
    can be parallel train pipeline use function Config.set_train_type()

    the returned train pipeline can be easily used by train(model,dataset),
    where model is obtained by Model.get_model(), dataset is obtained by Dataset.get_dataset()

    the train pipeline will:
    1.store and restore ckpt in directory ./save_dir/model_name/model_dir
    2.log loss information in directory ./save_dir/model_name/log.txt
    3.visualize model output periodly during training in directory ./save_dir/model_name/train_vis_dir
    the newest model is at path ./save_dir/model_name/model_dir/newest_model.npz

    Parameters
    ----------
    arg1 : config object
        the config object return by Config.get_config() function, which includes all the configuration information.
    
    Returns
    -------
    function
        a train pipeline function which takes model and dataset as input, can be either single train or parallel train pipeline.
    
    '''
    # determine train process
    model_type=config.model.model_type
    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose import single_train,parallel_train
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal import single_train,parallel_train
    else:
        raise RuntimeError(f'unknown model type {model_type}')
    print(f"training {model_type.name} model...")

    #determine train type
    train_type=config.train.train_type
    if(train_type==TRAIN.Single_train):
        train=partial(single_train,config=config)
    elif(train_type==TRAIN.Parallel_train):
        #set defualt kungfu opt type
        if("kungfu_option" not in config.train):
            config.train.kungfu_option=KUNGFU.Sma
        train=partial(parallel_train,config=config)
    print(f"using {train_type.name}...")
    return train

def get_evaluate(config):
    '''get evaluate pipeline based on config object

    construct evaluate pipeline based on the chosen model_type and dataset_type,
    the evaluation metric fellows the official metrics of the chosen dataset.

    the returned evaluate pipeline can be easily used by evaluate(model,dataset),
    where model is obtained by Model.get_model(), dataset is obtained by Dataset.get_dataset()

    the evaluate pipeline will:
    1.loading newest model at path ./save_dir/model_name/model_dir/newest_model.npz
    2.perform inference and parsing over the chosen evaluate dataset
    3.visualize model output in evaluation in directory ./save_dir/model_name/eval_vis_dir
    4.output model metrics by calling dataset.official_eval()

    Parameters
    ----------
    arg1 : config object
        the config object return by Config.get_config() function, which includes all the configuration information.
    
    Returns
    -------
    function
        a evaluate pipeline function which takes model and dataset as input, and output model metrics
    
    '''
    model_type=config.model.model_type
    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose import evaluate
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal import evaluate
    else:
        raise RuntimeError(f'unknown model type {model_type}')
    evaluate=partial(evaluate,config=config)
    print(f"evaluating {model_type.name} model...")
    return evaluate

def get_preprocess(model_type):
    '''get preprocess function based model_type

    get the preprocess function of the specified kind of model to help user construct thier own train
    and evaluate pipeline rather than using the integrated train or evaluate pipeline directly when in need.

    the preprocess function is able to convert the image and annotation to the model output format for training
    or evaluation.

    Parameters
    ----------
    arg1 : Config.MODEL
        a enum value of enum class Config.MODEL
    
    Returns
    -------
    function
        a preprocess function of the specified kind of model
    '''

    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose.utils import preprocess
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal.utils import preprocess
    return preprocess

def get_postprocess(model_type):
    '''get postprocess function based model_type

    get the postprocess function of the specified kind of model to help user construct thier own 
    evaluate pipeline rather than using the integrated train or evaluate pipeline directly when in need

    the postprocess function is able to parse the model output feature map and output parsed human objects of Human class,
    which contains all dectected keypoints.

    Parameters
    ----------
    arg1 : Config.MODEL
        a enum value of enum class Config.MODEL
    
    Returns
    -------
    function
        a postprocess function of the specified kind of model
    '''
    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose.utils import postprocess
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal.utils import postprocess
    return postprocess

def get_visualize(model_type):
    '''get visualize function based model_type

    get the visualize function of the specified kind of model to help user construct thier own 
    evaluate pipeline rather than using the integrated train or evaluate pipeline directly when in need

    the visualize function is able to visualize model's output feature map, which is helpful for
    training and evaluation analysis.

    Parameters
    ----------
    arg1 : Config.MODEL
        a enum value of enum class Config.MODEL
    
    Returns
    -------
    function
        a visualize function of the specified kind of model
    '''
    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose.utils import visualize
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal.utils import visualize
    return visualize