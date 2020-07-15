# Tutorial for Training Library
Up to now, Hyperpose provides:
* 4 types of preset model architectures:
    * Openpose 
    * LightweightOpenpose
    * Poseproposal
    * MobilenetThinOpenpose
* 7 types of common model backbone for backbone replacement:
    * MobilenetV1, MobilenetV2
    * Vggtiny, Vgg16, Vgg19
    * Resnet18, Resnet50
* 2 types of popular dataset
    * COCO
    * MPII

## Integrated pipeline
Hyperpose extract similiar models into a model class. For now, there are two classes: Openpose classes and Poseproposal classes.
all model architecture can be devided into one of them.  
For each model class, Hyperpose privide a integrated pipeline. 
### Integrated train pipeline
The usage of integrated training procedure of Hyperpose can be devided into two parts:  
setting configuration using APIs of *Config* module, and getting the configured system from the *Model* and *dataset* module.
* setting parts mainly concern:  model_name, model_type, model_backbone, dataset_type and train_type
    * *set_model_name* will determine what the path the model related file will be put to
    * *set_model_type* will adopt the chosen preset model architecture  
     (use enum value of enum class **Config.MODEL**)
    * *set_model_backbone* will replace the backbone of chosen preset model architeture  
     (use enum value of enum class **Config.BACKBONE**)
    * *set_dataset_type* will change the dataset in the training pipeline  
     (use enum value of enum class **Config.DATA**)
    * *set_train_type* is to choose whether use single GPU for single training or multiple GPUs for parallel training  
     (use enum value of enum class **Config.TRAIN**)<br>
the conbination of different model architectures and model backbones will lead to huge difference of countructed model' computation
complexity (for example,Openpose architecture with default Vgg19 backbone is 200MB, while MobilenetThinOpenpose with mobilenet-variant backbone is only 18MB), thus it should be carefully considered.  
    for more detailed information, please refer the API documents. 

The basic training pipeline configuration is below:
```bash
# >>> import modules of hyperpose
from hyperpose import Config,Model,Dataset
# >>> set model name is necessary to distinguish models (neccesarry)
Config.set_model_name(args.model_name)
# >>> set model architecture using Config.MODEL enum class (neccesarry)
Config.set_model_type(Config.MODEL.LightweightOpenpose)
# >>> set model backbone using Config.BACKBONE enum class (not neccessary, each model has its default backbone)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# >>> set dataset to use (neccesarry)
Config.set_dataset_type(Config.DATA.COCO)
# >>> set training type (not neccesary, default is single training, can use parallel training)
Config.set_train_type(Config.TRAIN.Single_train)
# >>> congratulations!, the simplest configuration is done, it's time to assemble the model and training pipeline
```
to use parallel training, one should set train type at first, and then choose kungfu optimizor wrap function, replace the set_train_type function as below, Kungfu also have three option: Sync_sgd,Sync_avg,Pair_avg
```bash
Config.set_train_type(Config.TRAIN.Parallel_train)
Config.set_kungfu_option(Config.KUNGFU.Sync_sgd)
```
And when run your program, using the following command(assuming we have 4 GPUs)
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 kungfu-run -np 4 python train.py
```

* getting parts mainly concern: pass the configuration to *Model* module and *Dataset* module to assemble the system
    * *Config.get_config* will return a config object which contains all the configuration and is the core of the getting functions 
    * *Model.get_model* will return a configrued model object which can forward and calcaulate loss
    * *Datset.get_dataset* will return a configured dataset object which can generate tensorflow dataset object used for train and evaluate, it can also visualize the dataset annotation.
    * *Model.get_train* will return a training pipeline, which could start running as long as receive the model object and dataset object

The basic training pipeline assembling is below:
```bash
# >>> get config object at first (neccesarry)
config=Config.get_config()
# >>> get model object (neccesarry)
model=Model.get_model(config)
# >>> get dataset object (neccesarry)
dataset=Dataset.get_dataset(config)
# >>> get train pipeline (neccesarry)
train=Model.get_train(config)
# >>> train!
train(model,dataset)
```

### Integrated evaluate pipeline
The configuration part and assembling part of evaluate pipeline is very similair to the train pipeline's.  
The only difference is that:
* In setting parts, one doesn't need to set train type(and thus kungfu option)
* In getting parts, one should use *Model.get_eval* to get evaluate pipeline(rather than train pipeline) 
Thus the evaluate code should be as follows:
```bash
# >>> import modules of hyperpose
from hyperpose import Config,Model,Dataset
# >>> set model name to be eval
Config.set_model_name(args.model_name)
# >>> the model architecture and backbone setting should be the same with the training configuration of the model to be evaluated.
Config.set_model_type(Config.MODEL.LightweightOpenpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# >>> set dataset to use
Config.set_dataset_type(Config.DATA.COCO)
# >>> configuration is done, get config object to assemble the system
config=Config.get_config()
model=Model.get_model(config)
dataset=Dataset.get_dataset(config)
eval=Model.get_eval(config)
# >>> eval
eval(model,dataset)
```
the model will be loaded from the ./save_dir/model_name/model_dir/newest_model.npz and evaluated.
It should be noted that:
* 1.the model architecture, model backbone, dataset type should be the same with the configuration under which model was trained.
* 2.the evaluation metrics will follow the official evaluation metrics of dataset

## User-defined model architecture
Hyperpose leaves freedom for user to define thier own model architecture but use the provided integrated model pipeline at the same time, the following points should be considered:<br>
* 1.the model should be an object of a tensorlayer.models.Model class (or inherite from this class)<br>
* 2.the model should have *foward* and *cal_loss* functions that has exactly the same input and output format with the preset model architectures. one can **refer Model.LightweightOpenpose class** for reference.<br>
to do this, user still need to set model_type to determine the training pipeline, here the model_type should be the model that has the similair data processing pipeline with the user's own model. Then he can use the *set_model_arch* function to pass
his own model object
```bash
    Config.set_model_name(your_model_name)
    Config.set_model_type(similiar_ model_type)
    Config.set_model_arch(your_model_arch)
```
the other configuration procedures are the same with the integrated training pipeline.

## User-defined dataset
Hyperpose allows user to use their own dataset to be integrated with the training and evaluating pipeline, as long as it has the following attribute and functions:<br>
* 1.**get_train_dataset**: <br>
return a tensorflow dataset object where each element should be a image path and a serialized dict(using **_pickle** library to serialize) which at least have the three key-value pairs: <br>
1.1 "kpt"-a list contains keyspoints for each labeled human, for example:[[kpt1,kpt2,...,kptn],[kpt1,kpt2,...,kptn]] is a list with two labeld humans, where each *kpt* is a [x,y] coordinate such as [234,526],etc<br>
1.2 "bbx"-a list contains boundingbox for each labeled human, for example:[bbx1,bbx2] is a list with two labeled humans, where each *bbx* is a [x,y,w,h] array such as [234,526,60,80], necessary for **pose proposal network**, could be set to *None* for others<br>
1.3 "mask"-a mask (in mscoco polynomial format) used to cover unlabeled people, could be set to *None*<br>
* 2.**get_eval_dataset**: <br>
return a tensorflow dataset object where each element should be a image path and its image id.<br>
* 3.**get_input_kpt_cvter**(optional): <br>
return a function which changes the **kpt** value in your dataset dict element,used to enable your dataset keypoint annotation being in line with your model keypoint setting, or combined with other dataset with different keypoint annotation.
* 4.**get_output_kpt_cvter**(optional): <br>
return a function which changes the model predict result to a format that easy to evaluate, used to enable your datatset to be evaluate at a formal COCO standard (using MAP) or MPII standard (using MPCH).  


## User-defined dataset filter
Hyperpose also leave freedom for user to define thier own dataset filter to filter the dataset as they want using *set_dataset_filter* function.
to use this, a user should know the follow information:
* 1.Hyperpose organize the annotations of one image in one dataset in the similiar meta classes.
for COCO dataset, it is COCOMeta; for MPII dataset, it is MPIIMeta.
Meta classes will have some common information such as image_id, joint_list etc,
they also have some dataset-specific imformation, such as mask, is_crowd, headbbx_list etc.
* 2.the dataset_fiter will perform on the Meta objects of the corresponding dataset, if 
it returns True, the image and annotaions the Meta object related will be kept,
otherwise it will be filtered out. Please refer the Dataset.xxxMeta classes for better use.
**please refer Dataset.COCOMeta,Dataset.MPIIMeta classes for better use.**
```bash
    def my_dataset_filter(coco_meta):
        if(len(coco_meta.joint_list)<5 and (not coco_meta.is_crowd)):
            return True
        else:
            return False
    Config.set_dataset_filter(my_dataset_filter)
```

## User-defined train pipeline
Hyperpose also provides three low level functions to help user consturct thier own pipeline. For each class of 
model, functions of preprocess, postprocess and visualize are provided.
* *get_preprocess* receives model_type from Config.MODEL and return a preprocess function
The preprocess function is able to convert the annotaion into targets used for training the model.
```bash
preprocess=Model.get_preprocess(Config.MODEL.LightweightOPenpose)
conf_map,paf_map=preprocess(annos,img_height,img_width,model_hout,model_wout,Config.DATA.COCO,data_format="channels_first")
pd_conf_map,pd_paf_map=my_model.forward(input_image[np.newaxis,...])
my_model.cal_loss(conf_map,pd_conf_map,paf_map,pd_paf_map)
```  

* *get_postprocess* receives model_type from Config.MODEL and return a postprocess function
The postprocess function is able to convert the model output into parsed human objects for evaluating and visualizing
```bash
pd_conf_map,pd_paf_map=my_model.forward(input_image[np.newaxis,...])
postprocess=Model.get_postprocess(Config.MODEL.LightweightOpenpose)
pd_humans=postprocess(pd_conf_map,pd_paf_map,dataset_type,data_format="channels_first")
for pd_human in pd_humans:
    pd_human.draw(input_image)
```  

* *get_visualize* receives model_type from Config.MODEL and return a visualize function
The visualize function is able visualize model's ouput feature map
```bash
pd_conf_map,pd_paf_map=my_model.forward(input_image[np.newaxis,...])
visualize=Model.get_visualize(Config.MODEL.LightweightOpenpose)
visualize(input_image,pd_conf_map,pd_paf_map,save_name="my_visual",save_dir="./vis_dir")
```

