# OpenPose using TensorFlow and TensorLayer

</a>
<p align="center">
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/dance_foot.gif?raw=true", width="360">
</p>


## 1. Motivation

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) from CMU provides real-time 2D pose estimation following ["Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields"](https://arxiv.org/pdf/1611.08050.pdf) However, the [training code](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) is based on Caffe and C++, which is hard to be customized.
While in practice, developers need to customize their training set, data augmentation methods according to their requirement.
For this reason, we reimplemented this project in [TensorLayer fashion](https://github.com/tensorlayer/tensorlayer).

🚀🚀 **This repo will be moved into [here](https://github.com/tensorlayer/tensorlayer/tree/master/examples) for life-cycle management soon. More cool Computer Vision applications such as super resolution and style transfer can be found in this [organization](https://github.com/tensorlayer).**

- TODO
  - [ ] Provides pretrained models
  - [ ] TensorRT Float16 and Int8 inference
  - [ ] Faster C++ post-processing
  - [ ] Distributed training
  - [ ] Faster data augmentation
  - [ ] Pose Proposal Networks, ECCV 2018

## 2. Project files

- `config.py` : config of the training details.
  -  set training mode : `datasetapi` (single gpu, default), `distributed` (multi-gpus, TODO), `placeholder `(slow, for debug only)
- `models.py`: defines the model structures.
- `utils.py`: utility functions.
- `train.py`: trains the model.

## 3. Preparation

Build C++ library for post processing. See: <https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess>

```bash
cd inference/pafprocess
make

# ** before recompiling **
rm -rf build
rm *.so
```

## 4. Train a model

Runs `train.py`, it will automatically download MSCOCO 2017 dataset into `dataset/coco17`.
The default model in `models.py` is based on VGG19, which is the same with the original paper.
If you want to customize the model, simply change it in `models.py`.
And then `train.py` will train the model to the end.

In `config.py`:

- `config.DATA.train_data` can be:
   * `coco_only`: training data is COCO dataset only (default)
   * `yours_only`: training data is your dataset specified by `config.DATA.your_xxx`
   * `coco_and_yours`: training data is COCO and your datasets

- `config.MODEL.name` can be:
   * `vgg`: VGG19 version (default), slow  
	* `vggtiny`: VGG tiny version, faster
	* `mobilenet`: MobileNet version, faster

- `config.TRAIN.train_mode` can be:
   * `datasetapi`: single GPU with TF dataset api pipeline (default)
   * `distributed`: multiple GPUs with TF dataset api pipeline, fast (you may need to change the hyper parameters according to your hardware)
   * `placeholder`: single GPU with placeholder, for debug only, slow

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

