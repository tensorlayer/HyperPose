# Tutorial for Training Library

## Overview

HyperPose Python Training library provides a one-step yet flexible platform for developing pose estimation models.

Based on the intended usage, there are two major categories of user requirements regarding developing a pose estimation system:

1. **Adapting existing algorithms to specific deployment scenarios**: e.g., select the pose estimation model architecture with best accuracy conditioned on the limited hardware resources.
2. **Developing customized pose estimation algorithms.**: e.g., explore new pose estimation algorithms with the help of existing dataset generation pipeline and model architectures. 

To meet these 2 kinds of user requirements, HyperPose provides both rich high-level APIs with integrated pipelines (for the first kind of requirement) and fine-grained APIs with in-depth customisation (for second kind of requirement). 

## Model/Algorithm/Dataset Supports

### 5 model algorithm classes

```{list-table} HyperPose preset model algorithms
:header-rows: 1

* - Algorithm
  - Class API
  - Description
* - [OpenPose](https://openaccess.thecvf.com/content_cvpr_2017/html/Cao_Realtime_Multi-Person_2D_CVPR_2017_paper.html)
  - `Openpose`
  - Original OpenPose algorithm.
* - [Light-weight OpenPose](https://arxiv.org/abs/1811.12004)
  - `LightweightOpenpose`
  - A light-weight variant of OpenPose with optimized prediction branch, designed for fast processing.
* - [MobileNet-Thin OpenPose](https://github.com/jiajunhua/ildoonet-tf-pose-estimation)
  - `MobilenetThinOpenpose`
  - A light-weight variant of OpenPose with adapted MobilNet backbone, featured with fast inference.
* - [PoseProposal](https://openaccess.thecvf.com/content_ECCV_2018/html/Sekii_Pose_Proposal_Networks_ECCV_2018_paper.html)
  - `Poseproposal`
  - A pose estimation algorithm which models key point detection as object detection with bounding box, featured with fast inference and post-processing.
* - [PifPaf](https://openaccess.thecvf.com/content_CVPR_2019/html/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.html)
  - `Pifpaf`
  - An accurate pose esitmation algorithm that generates high resolution confidence map, featured with high accuracy over low resolution images.
```

### 10 model backbone classes

```{list-table} HyperPose preset model backbones
:header-rows: 1
* - Model family
  - Backbone class APIs
* - Vgg Backbones
  - `vgg16_backbone`, `vgg19_backbone`, `vgg19_backbone`
* - Resnet Backbones
  - `Resnet18_backbone`, `Resnet18_backbone`
* - Mobilnet Backbones
  - `MobilenetV1_backbone`, `MobilenetV2_backbone`
```

### 2 dataset classes

```{list-table} HyperPose preset dataset
:header-rows: 1

* - Dataset name
  - Version
  - Size
* - [COCO](https://cocodataset.org/#home)
  - * COCO 2014 (version published in 2014)
    * COCO 2017 (version published in 2017)
  - 11'827 train images, 5'000 validation images, 40'670 test images

* - [MPII](http://human-pose.mpi-inf.mpg.de/)
  - MPII 2014 (version published in 2014)
  - 25'000 train images, 3'000 validation images, 7'000 test images.
```

### Training Options

- Parallel Training
- Backbone Pretraining
- Domain Adaptation

### Extensions

- Customized dataset
  - User-supplemented dataset

    User may supplement their self-collected data into preset dataset generation pipeline for training and evaluation.

  - User-defined dataset

    User may define their own dataset class to take over the whole dataset generation pipeline.

- Customized model

    User define thier own model class to take over the model forwarding and loss calculation procedure.

- Customized pipeline

    User use the provided pre-processors, post-processors and visualizers to assemble their own training or evalution pipeline.


## Integrated pipeline
HyperPose integrates training, evaluating and testing pipeline with various High-level APIs for quickly adapting the existing pose estimation algorithms for their customized usage scenarios.

The whole procedure can be devided into two parts:

**In the first part**, users use the `set` APIs of the **Config** module to set up the components of the pipeline. User can set up from the general settings such as algorithm type, network architecture and dataset type, to the detailed settings including training batch size, save interval and learning rate etc.

**In the second part**, users use the `get` APIs of the **Model** module and the **Dataset** module to assemble the system. After the configuration is finished, user could get a *config* object containing all the configuration. Pass the *config* object to the `get` APIs, users get the configured model, dataset, and the train or evaluate pipeline. 


The critical `set` APIs are below: (**necessary**)

- **Config.set_model_name**
 
  Receive a string, which is used to uniquely identify the model with a name(string).
  
  Model name is important as all the files generated during train, evaluate and test procedure of the specific model will be stored at **./save_dir/${MODEL_NAME}** directory. 

  Precisely, the related paths of the model with name of ${MODEL_NAME} are below:
  ```{list-table} Related paths to store files of the specific model.
  :header-rows: 1

  * - Folder Name
    - Path to what
  * - `./save_dir/${MODEL_NAME}/model_dir`
    - Model checkpoints.
  * - `./save_dir/${MODEL_NAME}/train_vis_dir`
    - Directory to save the train visualization images.
  * - `./save_dir/${MODEL_NAME}/eval_vis_dir`
    - Directory to save the evaluate visualization images.
  * - `./save_dir/${MODEL_NAME}/test_vis_dir`
    - Directory to save the test visualization images.
  * - `./save_dir/${MODEL_NAME}/data_vis_dir`
    - Directory to save the dataset label visualization images.
  * - `./save_dir/${MODEL_NAME}/frozen_${MODEL_NAME}.pb`
    - The default path to save the exported ProtoBuf format model. 
  * - `./save_dir/${MODEL_NAME}/log.txt`
    - The default path to save the training logs (e.g., loss).
  ```

- **Config.set_model_type**

  Receive an Enum value from **Config.MODEL** , which is used to determine the algorithm type to use.

  Available options are:
  ```{list-table} Available options of *Config.set_model_type*
  :header-rows: 1

  * - Available option
    - Description
  * - Config.MODEL.OpenPose
    - OpenPose algorithm
  * - Config.MODEL.LightweightOpenpose
    - Lightweight OpenPose algorithm
  * - Config.MODEL.MobilnetThinOpenpose
    - MobilenetThin OpenPose algorithm
  * - Conifg.MODEL.Poseproposal
    - Pose Proposal Network algorithm
  * - Config.MODEL.Pifpaf
    - Pifpaf algorithm
  ```

- **Config.set_model_backbone**

  Receive an Enum value from **Config.BACKBONE** , which is used to determine the network backbone to use.

  Different backbones will result in huge difference of the required computation resources. Each algorithm type has a default model backbone, while HyperPose also provides other backbones for replacement.

  Available options are:
  ```{list-table} Available options of *Config.set_model_backbone*
  :header-rows: 1

  * - Available option
    - Description
  * - Config.BACKBONE.Default
    - Use the default backbone of the preset algorithm
  * - Config.BACKBONE.Vggtiny
    - Adapted Vggtiny backbone
  * - Config.BACKBONE.Vgg16
    - Vgg16
  * - Config.BACKBONE.Vgg19
    - Vgg19
  * - Config.BACKBONE.Resnet18
    - Resnet18
  * - Config.BACKBONE.Resnet50
    - Resnet50
  * - Config.BACKBONE.Mobilenetv1
    - Mobilenetv1
  * - Config.BACKBONE.Mobilenetv2
    - Mobilenetv2
  ```

- **Config.set_dataset_type**

  Receive an Enum value from **Config.DATA** , which is used to determine the dataset to use.

  Different dataset will result in different train and evalution images and different evaluation metrics.

  Available options are:
  ```{list-table} Available options of *Config.set_model_backbone*
  :header-rows: 1

  * - Available option
    - Description
  * - Config.DATA.COCO
    - [Mscoco dataset](https://cocodataset.org/#home) 
  * - Config.DATA.MPII
    - [MPII dataset](http://human-pose.mpi-inf.mpg.de/)
  * - Config.DATA.USERDEF
    - Use user defined dataset.
  ```      

Use the necessary `set` APIs above, the basic model and dataset configuration is done, users can get the *config* object which contains all the configurations using the **Config.get_config** API:

- **Config.get_config**

  Receive nothing, return the *config* object.

Then user can get the *model* and *dataset* object for either train or evaluating using the `get` APIs.

The critical **Get** APIs are below:

- **Model.get_model**

  Receive the *config* object and return a configured *model* object.

  The *model* object comes from the *Tensorlayer.Model* class and should have the following functions:
  
  ```{list-table} Functions of the *model* object
  :header-rows: 1

  * - Function name
    - Function utility
  * - `forward`
    - Input image.

      Return predicted heat map.

  * - `cal_loss`
    - Input predicted heat map and ground truth heat map. 

      Return calcuated loss value.

  * - `save_weights`
    - Input save path and save format.

      Save the trained model weight.
  ```

- **Dataset.get_dataset**

	Receive the *config* object and return a configured *dataset* object.

	The *dataset* object should have the following functions:

  ```{list-table} Functions of the *dataset* object
  :header-rows: 1

  * - Function name
    - Function utility
  * - `get_parts`
    - Input nothing.

      Return pre-defined keypoints in the format of an *Enum* object.

  * - `get_colors`
    - Input nothing.

      Return pre-defined corresponding colors of keypoints in the format of a list.

  * - `generate_train_data`
    - Input nothing.

      Return a train image path list and the corresponding target list. Each target in the target list is 
      a *dict* object with keys of *kpt* (keypoint), *mask* (mask of unwanted image area) and *bbx* (keypoint bounding box) 

  * - `generate_eval_data`
    - Input nothing.

      Return a eval image path list and the corresponding image-id list.

  * - `generate_test_data`
    - Input nothing.

      Return a test image path list and the corresponding image-id list.
  ```

No matter using the train or evaluate pipeline, the above `set` and `get` process is always neccesarry to get the specific *model* and *config* object.  

User can either assemble thier own pipeline use the *model* and *dataset* object at hand, or they can use more fine-grained APIs to control the train and evaluate pipeline before use the **Config.get_config** API, so that they could use the *config* object to obain preset integrated train and evaluate pipeline for easy development.

How to use the Integrated train and evaluate pipeline are below.

### Integrated train pipeline
As mentioned above, the above usage of `set` and `get` APIs to get the *model* and *dataset* object are always necessary in HyperPose. 

To use the integrated train pipeline, the extra configuration APIs provided are below: (**Unnecessary for integrated train**)

- **Config.set_batch_size**
  
  Receive a integer, which is used as batch size in train procedure.

- **Config.set_learning_rate**

  Receive a floating point, which is used as the learning arte in train procedure.

- **Config.set_log_interval**

  Receive a integer, which is used as the interval bwteen logging loss information.

- **Config.set_train_type**

  Receive an Enum value from **Config.TRAIN**, which is used to determine the parallel training strategy.

  Available options:
  - Config.TRAIN.Single_train

      Use single GPU for training.

  - Config.TRAIN.Parallel_train
	
	  Use mutiple GPUs for parallel training.(Using Kungfu distributed training library)

- **Config.set_kungfu_option**

  Receive an Enum value from **Config.KUNGFU**, which is used to determine the optimize startegy of parallel training.

  Available options:
  - Config.KUNGFU.Sync_sgd
  - Config.KUNGFU.Sync_avg
  - Config.KUNGFU.Pair_avg

Then we need to use the `get` API to get the **train pipeline** from the **model** module:(**necessary for integrated train**)

- **Model.get_train**

  Receive the *config* object, return a training function.

  The training function takes the *model* object and the *dataset* object and automatically start training.

The basic code to use the integrate training pipeline is below:

```python
# set the train configuartion using 'set' APIs
# import modules of hyperpose
from hyperpose import Config,Model,Dataset
# set model name
Config.set_model_name(args.model_name)
# set model architecture
Config.set_model_type(Config.MODEL.LightweightOpenpose)
# set model backbone
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# set dataset to use
Config.set_dataset_type(Config.DATA.COCO)
# set training type
Config.set_train_type(Config.TRAIN.Single_train)

# assemble the system using 'get' APIs
# get config object
config=Config.get_config()
# get model object
model=Model.get_model(config)
# get dataset object
dataset=Dataset.get_dataset(config)
# get train pipeline
train=Model.get_train(config)
# train!
train(model,dataset)
```

To enable the parallel training, install the **Kungfu** library according to the [installation guide](../install/training.md), and using the following command when run your program.

```bash
# Assuming we have 4 GPUs and train.py is the python script that contain your HyperPose code
CUDA_VISIBLE_DEVICES=0,1,2,3 kungfu-run -np 4 python train.py
```

### Integrated evaluate pipeline
The usage of the integrated evaluate pipeline is similar to the usage pf the integrated training pipeline.  

The differences is that we use `get` APIs to get the evaluate pipeline.

(Remeber, we still need the `set` and `get` APIs used to get the *model* and *dataset* object as in how we use the integrate train pipeline.)

The API that we use `get` API to get the evaluate pipeline is below:

- **Model.get_eval**

  Receive the *config* object, return a evaluate function.

  The evaluate function take the *model* object and the *dataset* object, and automatically start evaluating.

The basic code to use the integrate evaluate pipeline is below:

```python
# set the evaluate pipeline using 'set' APIs
# import modules of hyperpose
from hyperpose import Config,Model,Dataset
# set model name to be eval
Config.set_model_name(args.model_name)
# set the model architecture and backbone according to the training configuration of the model to be evaluated.
Config.set_model_type(Config.MODEL.LightweightOpenpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# set dataset to use
Config.set_dataset_type(Config.DATA.MSCOCO)

# assemble the system using 'get' APIs
# get config object
config=Config.get_config()
# get model object
model=Model.get_model(config)
# get dataset object
dataset=Dataset.get_dataset(config)
# get evaluate pipeline
eval=Model.get_eval(config)
# evaluate!
eval(model,dataset)
```

It should be noticed that:
- the model architecture, model backbone, dataset type should be the same with the configuration under which model was trained.
- the model to evaluate will be loaded from the *./save_dir/${MODEL_NAME}/model_dir/newest_model.npz*.
- the evaluation metrics will follow the official evaluation metrics of dataset.

## User-defined model architecture
HyperPose leaves freedom for user to define thier own model architecture but use the provided integrated model pipeline at the same time, the following points should be considered:

* 1.the model should be an object of a tensorlayer.models.Model class (or inherite from this class)

* 2.the model should have *foward* and *cal_loss* functions that has exactly the same input and output format with the preset model architectures. one can **refer Model.LightweightOpenpose class** for reference.

To do this, user still need to set model_type to determine the training pipeline, here the model_type should be the model that has the similair data processing pipeline with the user's own model. Then he can use the *set_model_arch* function to pass
his own model object

```bash
    Config.set_model_name(your_model_name)
    Config.set_model_type(similiar_ model_type)
    Config.set_model_arch(your_model_arch)
```
The other configuration procedures are the same with the integrated training pipeline.

## User-defined dataset
HyperPose allows user to use their own dataset to be integrated with the training and evaluating pipeline, as long as it has the following attribute and functions:

- **get_train_dataset**: 

  Return a tensorflow dataset object where each element should be a image path and a serialized dict(using **_pickle** library to serialize) which at least have the three key-value pairs:

  - "kpt": a list contains keyspoints for each labeled human, for example:[[kpt1,kpt2,...,kptn],[kpt1,kpt2,...,kptn]] is a list with two labeld humans, where each *kpt* is a [x,y] coordinate such as [234,526],etc<br>

  - "bbx": a list contains boundingbox for each labeled human, for example:[bbx1,bbx2] is a list with two labeled humans, where each *bbx* is a [x,y,w,h] array such as [234,526,60,80], necessary for **pose proposal network**, could be set to *None* for others<br>

  - "mask": a mask (in mscoco polynomial format) used to cover unlabeled people, could be set to *None*<br>

- **get_eval_dataset**:

  Return a tensorflow dataset object where each element should be a image path and its image id.<br>

- **get_input_kpt_cvter**(optional):

  Return a function which changes the **kpt** value in your dataset dict element,used to enable your dataset keypoint annotation being in line with your model keypoint setting, or combined with other dataset with different keypoint annotation.

- **get_output_kpt_cvter**(optional): 

  Return a function which changes the model predict result to a format that easy to evaluate, used to enable your datatset to be evaluate at a formal COCO standard (using MAP) or MPII standard (using MPCH).  


## User-defined dataset filter
HyperPose also leave freedom for user to define thier own dataset filter to filter the dataset as they want using *set_dataset_filter* function.

To use this, a user should know the follow information:

- HyperPose organize the annotations of one image in one dataset in the similiar meta classes.

  For COCO dataset, it is COCOMeta; For MPII dataset, it is MPIIMeta.

  Meta classes will have some common information such as image_id, joint_list etc, they also have some dataset-specific imformation, such as mask, is_crowd, headbbx_list etc.

- The dataset_fiter will perform on the Meta objects of the corresponding dataset, if 
it returns True, the image and annotaions the Meta object related will be kept,
otherwise it will be filtered out. Please refer the Dataset.xxxMeta classes for better use.

**Please refer Dataset.COCOMeta,Dataset.MPIIMeta classes for better use.**

```python
    def my_dataset_filter(coco_meta):
        if(len(coco_meta.joint_list)<5 and (not coco_meta.is_crowd)):
            return True
        else:
            return False
    Config.set_dataset_filter(my_dataset_filter)
```

## User-defined train pipeline

HyperPose also provides three low level functions to help user consturct thier own pipeline. For each class of 
model, functions of preprocess, postprocess and visualize are provided.

- **Model.get_preprocess**
  Receive an Enum value from **Config.MODEL**, return a preprocess function.

  The preprocess function is able to convert the annotaion into targets used for training the model.

```python
preprocess=Model.get_preprocess(Config.MODEL.LightweightOPenpose)
conf_map,paf_map=preprocess(annos,img_height,img_width,model_hout,model_wout,Config.DATA.MSCOCO,data_format="channels_first")
pd_conf_map,pd_paf_map=my_model.forward(input_image[np.newaxis,...])
my_model.cal_loss(conf_map,pd_conf_map,paf_map,pd_paf_map)
```  

- **Model.get_postprocess** 
  Receive an Enum value from **Config.MODEL**, return a postprocess function.

  The postprocess function is able to convert the model output into parsed human objects for evaluating and visualizing.

```python
pd_conf_map,pd_paf_map=my_model.forward(input_image[np.newaxis,...])
postprocess=Model.get_postprocess(Config.MODEL.LightweightOpenpose)
pd_humans=postprocess(pd_conf_map,pd_paf_map,dataset_type,data_format="channels_first")
for pd_human in pd_humans:
    pd_human.draw(input_image)
```  

- **get_visualize**
  Receive an Enum value from **Config.MODEL**, return a visualize function.

  The visualize function is able visualize model's ouput feature map.

```python
pd_conf_map,pd_paf_map=my_model.forward(input_image[np.newaxis,...])
visualize=Model.get_visualize(Config.MODEL.LightweightOpenpose)
visualize(input_image,pd_conf_map,pd_paf_map,save_name="my_visual",save_dir="./vis_dir")
```

