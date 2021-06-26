import tensorflow as tf
from functools import partial
from .human import Human
from .common import rename_tensor
from .common import TRAIN,MODEL,DATA,KUNGFU,BACKBONE
from .openpose import OpenPose,LightWeightOpenPose,MobilenetThinOpenpose
from .pose_proposal import PoseProposal
from .backbones import MobilenetV1_backbone,MobilenetV2_backbone,vgg16_backbone,vgg19_backbone,vggtiny_backbone
from .backbones import Resnet18_backbone,Resnet50_backbone
from .pretrain import single_pretrain

#claim:
#all the model preprocessor,postprocessor,and visualizer processing logic are written in 'channels_first' data_format
#input data in "channels_last" data_format will be converted to "channels_first" format first and then handled

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
                print(f"using default model backbone!")
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
        pretraining=config.pretrain.enable
        print(f"enable model backbone pretraining:{pretraining}")
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
        elif(model_type == MODEL.Pifpaf):
            from  .pifpaf.utils import get_parts
            from  .pifpaf.utils import get_limbs
            model.parts=get_parts(dataset_type)
            model.limbs=get_limbs(dataset_type)

        userdef_parts=config.model.userdef_parts
        userdef_limbs=config.model.userdef_limbs
        if(userdef_parts!=None):
            print("using user-defined model parts!")
            model.parts=userdef_parts
        if(userdef_limbs!=None):
            print("using user-defined model limbs!")
            model.limbs=userdef_limbs
        
        #set model
        if model_type == MODEL.Openpose:
            from .openpose import OpenPose as model_arch
            ret_model=model_arch(parts=model.parts,n_pos=len(model.parts),limbs=model.limbs,n_limbs=len(model.limbs),num_channels=model.num_channels,\
                hin=model.hin,win=model.win,hout=model.hout,wout=model.wout,backbone=backbone,pretraining=pretraining,data_format=model.data_format)
        elif model_type == MODEL.LightweightOpenpose:
            from .openpose import LightWeightOpenPose as model_arch
            ret_model=model_arch(parts=model.parts,n_pos=len(model.parts),limbs=model.limbs,n_limbs=len(model.limbs),num_channels=model.num_channels,\
                hin=model.hin,win=model.win,hout=model.hout,wout=model.wout,backbone=backbone,pretraining=pretraining,data_format=model.data_format)
        elif model_type == MODEL.MobilenetThinOpenpose:
            from .openpose import MobilenetThinOpenpose as model_arch
            ret_model=model_arch(parts=model.parts,n_pos=len(model.parts),limbs=model.limbs,n_limbs=len(model.limbs),num_channels=model.num_channels,\
                hin=model.hin,win=model.win,hout=model.hout,wout=model.wout,backbone=backbone,pretraining=pretraining,data_format=model.data_format)
        elif model_type == MODEL.PoseProposal:
            from .pose_proposal import PoseProposal as model_arch
            ret_model=model_arch(parts=model.parts,K_size=len(model.parts),limbs=model.limbs,L_size=len(model.limbs),hnei=model.hnei,wnei=model.wnei,lmd_rsp=model.lmd_rsp,\
                lmd_iou=model.lmd_iou,lmd_coor=model.lmd_coor,lmd_size=model.lmd_size,lmd_limb=model.lmd_limb,backbone=backbone,\
                pretraining=pretraining,data_format=model.data_format)
        elif model_type == MODEL.Pifpaf:
            from .pifpaf import Pifpaf as model_arch
            ret_model=model_arch(parts=model.parts,n_pos=len(model.parts),limbs=model.limbs,n_limbs=len(model.limbs),hin=model.hin,win=model.win,\
                scale_size=32,pretraining=pretraining,data_format=model.data_format)
        else:
            raise RuntimeError(f'unknown model type {model_type}')
        print(f"using {model_type.name} model arch!")
    return ret_model

def get_pretrain(config):
    return partial(single_pretrain,config=config)

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
    elif model_type == MODEL.Pifpaf:
        from .pifpaf import single_train,parallel_train
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
    elif model_type == MODEL.Pifpaf:
        from .pifpaf import evaluate
    else:
        raise RuntimeError(f'unknown model type {model_type}')
    evaluate=partial(evaluate,config=config)
    print(f"evaluating {model_type.name} model...")
    return evaluate

def get_test(config):
    '''get test pipeline based on config object

    construct test pipeline based on the chosen model_type and dataset_type,
    the test metric fellows the official metrics of the chosen dataset.

    the returned test pipeline can be easily used by test(model,dataset),
    where model is obtained by Model.get_model(), dataset is obtained by Dataset.get_dataset()

    the test pipeline will:
    1.loading newest model at path ./save_dir/model_name/model_dir/newest_model.npz
    2.perform inference and parsing over the chosen test dataset
    3.visualize model output in test in directory ./save_dir/model_name/test_vis_dir
    4.output model test result file at path ./save_dir/model_name/test_vis_dir/pd_ann.json
    5.the test dataset ground truth is often preserved by the dataset creator, you may need to upload the test result file to the official server to get model test metrics

    Parameters
    ----------
    arg1 : config object
        the config object return by Config.get_config() function, which includes all the configuration information.
    
    Returns
    -------
    function
        a test pipeline function which takes model and dataset as input, and output model metrics
    
    '''
    model_type=config.model.model_type
    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose import test
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal import test
    elif model_type == MODEL.Pifpaf:
        from .pifpaf import test
    else:
        raise RuntimeError(f'unknown model type {model_type}')
    test=partial(test,config=config)
    print(f"testing {model_type.name} model...")
    return test

def get_preprocessor(model_type):
    '''get a preprocessor class based on the specified model_type

    get the preprocessor class of the specified kind of model to help user directly construct their own 
    train pipeline(rather than using the integrated train pipeline) when in need.

    the preprocessor class is able to construct a preprocessor object that could convert the image and annotation to 
    the model output format for training.

    Parameters
    ----------
    arg1 : Config.MODEL
        a enum value of enum class Config.MODEL
    
    Returns
    -------
    class
        a preprocessor class of the specified kind of model
    '''

    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose import PreProcessor
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal import PreProcessor
    elif model_type == MODEL.Pifpaf:
        from .pifpaf import PreProcessor
    return PreProcessor

def get_postprocessor(model_type):
    '''get a postprocessor class based on the specified model_type

    get the postprocessor class of the specified kind of model to help user directly construct their own 
    evaluate pipeline(rather than using the integrated evaluate pipeline) or infer pipeline(to check the model utility) 
    when in need.

    the postprocessor is able to parse the model output feature map and output parsed human objects of Human class,
    which contains all dectected keypoints.

    Parameters
    ----------
    arg1 : Config.MODEL
        a enum value of enum class Config.MODEL
    
    Returns
    -------
    function
        a postprocessor class of the specified kind of model
    '''
    if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
        from .openpose import PostProcessor
    elif model_type == MODEL.PoseProposal:
        from .pose_proposal import PostProcessor
    elif model_type == MODEL.Pifpaf:
        from .pifpaf import PostProcessor
    return PostProcessor

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