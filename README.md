# OpenPose-Plus: Fast and Flexible OpenPose Framework based on TensorFlow and TensorLayer

</a>
<p align="center">
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/dance_foot.gif?raw=true", width="360">
</p>


## Motivation

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is the state-of-the-art real-time 2D pose estimation algorithm. 
In the official Caffe-based [codebase](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation),
data processing, training, and neural network blocks are heavily interleaved and mostly hard-coded. This makes it difficult
to be customised for achieving the best performance in our custom pose estimation applications.
Hence, we develop OpenPose-Plus, a fast and flexible pose estimation framework that offers the following powerful features:
- Flexible combination of standard training dataset with your own custom labelled data.
- Customisable data augmentation pipeline without compromising performance
- Deployment on embedded platforms using TensorRT
- Switchable neural networks (e.g., changing VGG to MobileNet for minimal memory consumption)
- Integrated training on a single GPU and multiple GPUs

## Work in progress

This project is still under active development, some of the TODOs are as follows:
- Distributed training
- Pose Proposal Networks, ECCV 2018

## Key project files

- `config.py` : config of the training details.
- `train.py`: trains the model.

## Preparation

Build C++ library for post processing. See [instruction](https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess)

```bash
cd inference/pafprocess
make

# ** before recompiling **
rm -rf build
rm *.so
```

## Training your model

Training the model is implemented using TensorFlow. To run `train.py`, you would need to install packages, shown
in [requirements.txt](https://github.com/tensorlayer/openpose-plus/blob/master/requirements.txt), in your virtual environment (Python <=3.6):

```bash
pip install -r requirements.txt
pip install pycocotools
```

`train.py` will automatically download MSCOCO 2017 dataset into `dataset/coco17`.
The default model in `models.py` is based on VGG19, which is the same with the original paper.
If you want to customize the model, simply change it in `models.py`.
And then `train.py` will train the model to the end.

In `config.py`, `config.DATA.train_data` can be:
* `coco`: training data is COCO dataset only (default)
* `custom`: training data is your dataset specified by `config.DATA.your_xxx`
* `coco_and_custom`: training data is COCO and your dataset

`config.MODEL.name` can be:
* `vgg`: VGG19 version (default), slow  
* `vggtiny`: VGG tiny version, faster
* `mobilenet`: MobileNet version, faster

`config.TRAIN.train_mode` can be:
* `single`: single GPU training
* `parallel`: parallel GPU training (on-going work)

## 5. Inference

Currently we provide two C++ APIs for inference, both defined in `include/openpose-plus.hpp`.
They are for running the tensorflow model with tensorRT and post-processing respectively.

You can look the examples in the `examples` folder to see how to use the APIs.
Running `./scripts/live-camera.sh` will give you a quick review of how it works.

You can build the APIs into a standard C++ library by just running `make pack`, provided that you have the following dependencies installed

  - tensorRT
  - opencv

<!---
## 5. Inference

In this project, input images are RGB with 0~1.
Runs `train.py`, it will automatically download the default VGG19-based model from [here](https://github.com/tensorlayer/pretrained-models), and use it for inferencing.
The performance of pre-trained model is as follow:


|                  | Speed | AP | xxx |
|------------------|-------|----|-----|
| VGG19            | xx    | xx | xx  |
| Residual Squeeze | xx    | xx | xx  |

- Speed is tested on XXX

- We follow the [data format of official OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md)

To use the pre-trained models

-->


<!--
## 6. Evaluate a model

Runs `eval.py` for inference.


## . Speed up and deployment

For TensorRT float16 (half-float) inferencing, xxx


## 6. Customization

- Model : change `models.py`.
- Data augmentation : change `train.py`
- Train with your own data: ....
    1. prepare your data following MSCOCO format, you need to .
    2. concatenate the list of your own data JSON into ...
-->

## Discussion

- [TensorLayer Slack](https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc)
- [TensorLayer WeChat](https://github.com/tensorlayer/tensorlayer-chinese/blob/master/docs/wechat_group.md)
- [TensorLayer Issues 434](https://github.com/tensorlayer/tensorlayer/issues/434)
- [TensorLayer Issues 416](https://github.com/tensorlayer/tensorlayer/issues/416)

<!--
## Paper's Model

- [Default MPII](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_MPI/pose_deploy.prototxt)
- [Default COCO model](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_COCO/pose_deploy.prototxt)
- [Visualizing Caffe model](http://ethereon.github.io/netscope/#/editor)
-->
