import logging
from functools import partial
from .common import TRAIN,MODEL,DATA,KUNGFU,BACKBONE
from .backbones import MobilenetV1_backbone
from .backbones import MobilenetV2_backbone
from .backbones import MobilenetDilated_backbone
from .backbones import MobilenetThin_backbone
from .backbones import MobilenetSmall_backbone
from .backbones import vggtiny_backbone
from .backbones import vgg16_backbone
from .backbones import vgg19_backbone
from .backbones import Resnet18_backbone
from .backbones import Resnet50_backbone
from .pretrain import single_pretrain
from .common import log_model as log
from .augmentor import BasicAugmentor
from .examine import exam_model_weights, exam_npz_dict_weights, exam_npz_weights
from .processor import ImageProcessor

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
        log("Using user self-defined model arch!")
        ret_model=model.model_arch(config)
    #using defualt arch
    else:
        backbone=None
        if("model_backbone" in model):
            model_backbone=model.model_backbone
            if(model_backbone==BACKBONE.Default):
                log(f"Using default model backbone!")
            elif(model_backbone==BACKBONE.Mobilenetv1):
                backbone=MobilenetV1_backbone
                log(f"Setting MobilnetV1_backbone!")
            elif(model_backbone==BACKBONE.Mobilenetv2):
                backbone=MobilenetV2_backbone
                log(f"Setting MobilenetV2_backbone!")
            elif(model_backbone==BACKBONE.MobilenetDilated):
                backbone=MobilenetDilated_backbone
                log(f"Setting MobilenetDilated_backbone!")
            elif(model_backbone==BACKBONE.MobilenetThin):
                backbone=MobilenetThin_backbone
                log(f"Setting MobilenetThin_backbone!")
            elif(model_backbone==BACKBONE.MobilenetSmall):
                backbone=MobilenetSmall_backbone
                log("Setting MobilenetSmall_backbone!")
            elif(model_backbone==BACKBONE.Vggtiny):
                backbone=vggtiny_backbone
                log(f"Setting Vggtiny_backbone!")
            elif(model_backbone==BACKBONE.Vgg16):
                backbone=vgg16_backbone
                log(f"Setting Vgg16_backbone!")
            elif(model_backbone==BACKBONE.Vgg19):
                backbone=vgg19_backbone
                log(f"Setting Vgg19_backbone!")
            elif(model_backbone==BACKBONE.Resnet18):
                backbone=Resnet18_backbone
                log(f"Setting Resnet18_backbone!")
            elif(model_backbone==BACKBONE.Resnet50):
                backbone=Resnet50_backbone
                log(f"Setting Resnet50_backbone!")
            else:
                raise NotImplementedError(f"Unknown model backbone {model_backbone}")

        model_type=model.model_type
        dataset_type=config.data.dataset_type
        pretraining=config.pretrain.enable
        log(f"Enable model backbone pretraining:{pretraining}")
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

        custom_parts=config.model.custom_parts
        custom_limbs=config.model.custom_limbs
        if(custom_parts!=None):
            log("Using user-defined model parts")
            model.parts=custom_parts
        if(custom_limbs!=None):
            log("Using user-defined model limbs")
            model.limbs=custom_limbs
        
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
        log(f"Using {model_type.name} model arch!")
    info_propt()
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
    model_config = config.model
    model_type = config.model.model_type
    train_type = config.train.train_type
    # determine train type
    if(train_type == TRAIN.Single_train):
        from .train import single_train as train_pipeline
        log("Single train procedure initialized!")
    elif(train_type == TRAIN.Parallel_train):
        from .train import parallel_train as train_pipeline
        log("Parallel train procedure initialized!")
    # get augmentor
    Augmentor = get_augmentor(config)
    augmentor = Augmentor(**model_config)
    log("Augmentor initialized!")
    # get preprocessor
    PreProcessor = get_preprocessor(config)
    preprocessor = PreProcessor(**model_config)
    log("Preprocessor initialized!")
    # get postprocessor
    PostProcessor = get_postprocessor(config)
    postprocessor = PostProcessor(**model_config)
    log("Postprocessor initialized!")
    # get visualizer
    Visualizer = get_visualizer(config)
    visualizer = Visualizer(save_dir=config.train.vis_dir,**model_config)
    log("Visualizer initialized!")
    
    # assemble training pipeline
    train = partial(
        train_pipeline,
        config = config,
        augmentor = augmentor,
        preprocessor = preprocessor,
        postprocessor = postprocessor,
        visualizer = visualizer
    )
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
    log(f"evaluating {model_type.name} model...")
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
    log(f"testing {model_type.name} model...")
    return test

def get_augmentor(config):
    if(config.model.custom_augmentor is not None):
        return config.model.custom_augmentor
    else:
        return BasicAugmentor

def get_preprocessor(config):
    '''get a preprocessor class based on the specified model_type

    get the preprocessor class of the specified kind of model to help user directly construct their own 
    train pipeline(rather than using the integrated train pipeline) when in need.

    the preprocessor class is able to construct a preprocessor object that could convert the image and annotation to 
    the model output format for training.

    Parameters
    ----------
    arg1 : config
        config object return by Config.get_config function
    
    Returns
    -------
    class
        a preprocessor class of the specified kind of model
    '''
    model_type = config.model.model_type
    if(config.model.custom_preprocessor is not None):
        return config.model.custom_preprocessor
    else:
        if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
            from .openpose import PreProcessor
        elif model_type == MODEL.PoseProposal:
            from .pose_proposal import PreProcessor
        elif model_type == MODEL.Pifpaf:
            from .pifpaf import PreProcessor
        return PreProcessor

def get_postprocessor(config):
    '''get a postprocessor class based on the specified model_type

    get the postprocessor class of the specified kind of model to help user directly construct their own 
    evaluate pipeline(rather than using the integrated evaluate pipeline) or infer pipeline(to check the model utility) 
    when in need.

    the postprocessor is able to parse the model output feature map and output parsed human objects of Human class,
    which contains all dectected keypoints.

    Parameters
    ----------
    arg1 : config
        config object return by Config.get_config function
    
    Returns
    -------
    function
        a postprocessor class of the specified kind of model
    '''
    model_type = config.model.model_type
    if(config.model.custom_postprocessor is not None):
        return config.model.custom_postprocessor
    else:
        if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
            from .openpose import PostProcessor
        elif model_type == MODEL.PoseProposal:
            from .pose_proposal import PostProcessor
        elif model_type == MODEL.Pifpaf:
            from .pifpaf import PostProcessor
        return PostProcessor

def get_visualizer(config):
    '''get visualize function based model_type

    get the visualize function of the specified kind of model to help user construct thier own 
    evaluate pipeline rather than using the integrated train or evaluate pipeline directly when in need

    the visualize function is able to visualize model's output feature map, which is helpful for
    training and evaluation analysis.

    Parameters
    ----------
    arg1 : config
        config object return by Config.get_config function
    
    Returns
    -------
    function
        a visualize function of the specified kind of model
    '''
    model_type = config.model.model_type
    if(config.model.custom_visualizer is not None):
        return config.model.custom_visualizer
    else:
        if model_type == MODEL.Openpose or model_type == MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose:
            from .openpose import Visualizer
        elif model_type == MODEL.PoseProposal:
            from .pose_proposal import Visualizer
        elif model_type == MODEL.Pifpaf:
            from .pifpaf import Visualizer
        return Visualizer

def get_imageprocessor():
    return ImageProcessor

def info(msg):
    info_logger = logging.getLogger("INFO")
    info_logger.info(msg)

def info_propt():
    # information propt
    print("\n")
    info("Welcome to Hyperpose Development Platform!")
    print("\n"+"="*100)
    
    # variable definition
    info("Variable Definition:")

    info("parts: \t the joints of human body, Enum class")
    info("limbs: \t the limbs of human body, List of tuple.\t example: [(joint index 1, joint index 2),...]")
    info("colors:\tthe visualize color for each parts, List.\t example: [(0,0,255),...] (optional)")

    info("n_parts:\tnumber of human body joints, int.\t example: n_parts=len(parts)")
    info("n_limbs:\tnumber of human body limbs, int.\t example: n_limbs=len(limbs)")

    info("hin: \t height of the model input image, int.\t example: 368")
    info("win: \t width of the model input image, int.\t example: 368")
    info("hout: \t height of model output heatmap, int.\t example: 46")
    info("wout: \t wout of model output heatmap, int.\t example: 46")
    print("\n"+"="*100)

    # object definition
    info("Object Definition:")
    info("config: a object contains all the configurations used to assemble the model, dataset, and pipeline. easydict object.\n"+\
            "\t return by the `Config.get_config` function.\n")

    info("model: a neural network takes in the image and output the calculated activation map. BasicModel object.\n"\
            +"\t have `forward`, `cal_loss`, `infer`(optional) functions.\n"
            +"\t custom: users could inherit the Model.BasicModel class for customization.\n"
            +"\t example: please refer to Model.LightWeightOpenPose class for details. \n")

    info("dataset: a dataset generator provides train and evaluate dataset. Base_dataset object.\n"\
            +"\t have `get_train_dataset` and `get_eval_dataset` functions.\n" \
            +"\t custom: users could inherit the Dataset.BasicDataset class for customizationn\n"
            +"\t example: please refer to Datset.CocoDataset class for details.\n")
            
    info("augmentor: a data augmentor that takes in the image, key point annotation, mask and perform affine transformation "\
            +"for data augmentation. BasicAumentor object.\n"\
            +"\t have `process` and `process_only_image` functions.\n"
            +"\t custom: users could inherit the Model.BasicAugmentor class for customization.\n"
            +"\t example: please refer to Model.BasicAugmentor class for details.\n")

    info("preprocessor: a data preprocessor that takes in the image, key point annotation and mask to produce the target heatmap\n"\
            +"\tfor model to calculate loss and learn. BasicPreProcessor object.\n"\
            +"\t have `process` function.\n"
            +"\t custom: users could inherit the Model.BasicPreProcessor class for customizationn\n"
            +"\t example: please refer to Model.openpose.PreProcessor class for details.\n")
    
    info("postprocessor: a data postprocessor that takes in the predicted heatmaps and infer the human body joints and limbs.\n"\
            +"\t have `process` function. BasicPostProcessor object.\n"
            +"\t custom: users could inherit the Model.BasicPostProcessor class for customization\n"
            +"\t example: please refer to the Model.openpose.PostProcessor class for details.\n")

    info("visualizer: a visualizer that takes in the predicted heatmaps and output visualization images for train and evaluation.\n"\
            +"\t have `visualize` and `visualize_comapre` functions. BasicVisualizer object.\n"\
            +"\t custom: users could inherit the Model.BasicVisualizer class for customization.\n"
            +"\t example: please refer to the Model.openpose.Visualizer class for details.\n"
            )
    print("\n"+"="*100)

    info("Development platform basic usage:\n"\
            +"\t1.Use the `sets` APIs of Config module to configure the pipeline, choose the algorithm type, the neural network\n"\
                +"\tbackbone, the dataset etc. that best fit your application scenario.\n"\
            +"\t2.Use the `get_model` API of Model module to get the configured model, use `get_dataset` API of dataset module to\n"\
                +"\tget the configured dataset, use the `get_train` API of Model module to get the configured train procedure. Then start\n"\
                +"\ttraining! Check the loss values and sample training result images during training.\n"
            +"\t3.Use the `get_eval` API of Model module to get the configured evaluation procedure. evaluate the model you trained. \n"\
            +"\t4.Eport the model to .pb, .onnx, .tflite formats for deployment."
        )
    
    info("Development platform custom usage:\n"\
            +"\t Hyperpose enables users to custom model, dataset, augmentor, preprocessor, postprocessor and visualizer.\n"\
            +"\t Users could inherit the corresponding basic class(mentioned above), and implement corresponding the member functions\n"\
                +"\trequired according to the function annotation, then use Config.set_custom_xxx APIs to set the custom component.")
            
    info("Additional features:\n"\
            +"\t 1.Parallel distributed training with Kungfu.\n"
            +"\t 2.Domain adaption to leverage unlabled data.\n"
            +"\t 3.Neural network backbone pretraining.")
    
    info("Currently all the procedures are uniformed to be `channels_first` data format.")
    info("Currently all model weights are saved in `npz_dict` format.")
    print("\n"+"="*100)