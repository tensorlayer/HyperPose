import os
from .common import TRAIN,MODEL,DATA
from .base_dataset import Base_dataset
from .multi_dataset import Multi_dataset
from .mpii_dataset import MPII_dataset
from .mscoco_dataset import MSCOCO_dataset
from .imagenet_dataset import Imagenet_dataset
from .common import imread_rgb_float,imwrite_rgb_float

def get_dataset(config):
    '''get dataset object based on the config object

    consturct and return a dataset object based on the config.
    No matter what the bottom dataset type is, the APIs of the returned dataset object are uniform, they are 
    the following APIs:

        visualize: visualize annotations of the train dataset and save it in "data_vir_dir"
        get_dataset_type: return the type of the bottom dataset.
        get_train_dataset: return a uniform tensorflow dataset object for training.
        get_val_dataset: return a uniform tensorflow dataset object for evaluating.
        official_eval: perform official evaluation on this dataset.


    The construction pipeline of this dataset object is below:

        1.check whether the dataset file(official zip or mat) is under data_path,
        if it isn't, download it from official website automaticly

        2.decode the official dataset file, organize the annotations in corresponding Meta classes,
        conveniet for processing.

        3.based on annotation, split train and evaluat part for furthur use.

    if user defined thier own dataset_filter, it will be executed in the train dataset or evaluate dataset generating procedure.

    use the APIs of this returned dataset object, the difference of different dataset is minimized.

    Parameters
    ----------
    arg1 : config object
        the config object return by Config.get_config() function, which includes all the configuration information.
    
    Returns
    -------
    dataset
        a dataset object with unifrom APIs:
        visualize, get_dataset_type, get_train_dataset, get_val_dataset,official_eval
    '''
    model_type=config.model.model_type
    dataset_type=config.data.dataset_type
    if(dataset_type==DATA.MSCOCO):
        print("using MSCOCO dataset!")
        if(model_type==MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose or model_type==MODEL.Openpose):
            from .mscoco_dataset.define import opps_input_converter as input_kpt_cvter
            from .mscoco_dataset.define import opps_output_converter as output_kpt_cvter
        elif(model_type==MODEL.PoseProposal):
            from .mscoco_dataset.define import ppn_input_converter as input_kpt_cvter
            from .mscoco_dataset.define import ppn_output_converter as output_kpt_cvter
        elif(model_type==MODEL.Pifpaf):
            from .mscoco_dataset.define import pifpaf_input_converter as input_kpt_cvter
            from .mscoco_dataset.define import pifpaf_output_converter  as output_kpt_cvter
        dataset=MSCOCO_dataset(config,input_kpt_cvter,output_kpt_cvter)
        dataset.prepare_dataset()
    elif(dataset_type==DATA.MPII):
        print("using MPII dataset!")
        if(model_type==MODEL.LightweightOpenpose or model_type==MODEL.MobilenetThinOpenpose or model_type==MODEL.Openpose):
            from .mpii_dataset.define import opps_input_converter as input_kpt_cvter
            from .mpii_dataset.define import opps_output_converter as output_kpt_cvter
        elif(model_type==MODEL.PoseProposal):
            from .mpii_dataset.define import ppn_input_converter as input_kpt_cvter
            from .mpii_dataset.define import ppn_output_converter as output_kpt_cvter
        dataset=MPII_dataset(config,input_kpt_cvter,output_kpt_cvter)
        dataset.prepare_dataset()
    elif(dataset_type==DATA.USERDEF):
        print("using user-defined dataset!")
        userdef_dataset=config.data.userdef_dataset
        dataset=userdef_dataset(config)
    elif(dataset_type==DATA.MULTIPLE):
        print("using multiple-combined dataset!")
        combined_dataset_list=[]
        multiple_dataset_configs=config.data.multiple_dataset_configs
        print(f"total {len(multiple_dataset_configs)} datasets settled, initializing combined datasets individualy....")
        for dataset_idx,dataset_config in enumerate(multiple_dataset_configs):
            print(f"initializing combined dataset {dataset_idx},config:{dataset_config.data}...")
            combined_dataset_list.append(get_dataset(dataset_config))
        print("initialization finished")
        dataset=Multi_dataset(config,combined_dataset_list)
    else:
        raise NotImplementedError(f"invalid dataset_type:{dataset_type}")
    return dataset


def get_pretrain_dataset(config):
    return Imagenet_dataset(config)

def enum2dataset(dataset_type):
    if(dataset_type==DATA.MSCOCO):
        return MSCOCO_dataset
    elif(dataset_type==DATA.MPII):
        return MPII_dataset
    elif(dataset_type==DATA.MULTIPLE):
        raise NotImplementedError("multiple dataset shouldn't be nested!")
    else:
        raise NotImplementedError("Unknow dataset!")
