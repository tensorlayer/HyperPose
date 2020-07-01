import os
from .common import TRAIN,MODEL,DATA
from .mpii_dataset.dataset import MPII_dataset
from .mscoco_dataset.dataset import MSCOCO_dataset

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

    dataset_type=config.data.dataset_type
    if(dataset_type==DATA.MSCOCO):
        print("using Mscoco dataset!")
        dataset=MSCOCO_dataset(config)
        dataset.prepare_dataset()
    elif(dataset_type==DATA.MPII):
        print("using Mpii dataset!")
        dataset=MPII_dataset(config)
        dataset.prepare_dataset()
    else:
        print("using user-defined dataset!")
        user_dataset=dataset_type
        dataset=user_dataset(config)
    return dataset
