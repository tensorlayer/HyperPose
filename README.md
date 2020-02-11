# OpenPose-Plus: Pose Estimation in the Wild

</a>
<p align="center">
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/dance_foot.gif?raw=true", width="360">
</p>

[![Documentation Status](https://readthedocs.org/projects/openpose-plus/badge/?version=latest)](https://openpose-plus.readthedocs.io/en/latest/?badge=latest)

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is the state-of-the-art pose estimation algorithm.
In its Caffe [codebase](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation),
data augmentation, training, and neural networks are most hard-coded. They are difficult
to be customized. In addition,
key performance features such as embedded platform supports and parallel GPU training are missing.
All these limitations makes OpenPose, in these days, hard to
be deployed in the wild. To resolve this, we develop **OpenPose-Plus**, a high-performance yet flexible pose estimation framework that offers many powerful features:
- Flexible combination of standard training dataset with your own custom labelled data.
- Customizable data augmentation pipeline without compromising performance
- Deployment on embedded platforms using TensorRT
- Switchable neural networks (e.g., changing VGG to MobileNet for minimal memory consumption)
- High-performance training using multiple GPUs

## Custom Model Training

Training the model is implemented using TensorFlow. To run `train.py`, you would need to install packages, shown
in [requirements.txt](https://github.com/tensorlayer/openpose-plus/blob/master/requirements.txt), in your virtual environment (**Python 3**):

```bash
pip3 install -r requirements.txt
pip3 install pycocotools
```

`train.py` automatically download MSCOCO 2017 dataset into `dataset/coco17`.
The default model is VGG19 used in the OpenPose paper.
To customize the model, simply changing it in `models.py`.

You can use `train_config.py` to configure the training. `config.DATA.train_data` can be:
* `coco`: training data is COCO dataset only (default)
* `custom`: training data is your dataset specified by `config.DATA.your_xxx`
* `coco_and_custom`: training data is COCO and your dataset

`config.MODEL.name` can be:
* `vgg`: VGG19 version (default), slow
* `vggtiny`: VGG tiny version, faster
* `mobilenet`: MobileNet version, faster

Train your model by running:

```bash
python3 train.py
```

### Additional steps for training on Windows

There are a few extra steps to follow with Windows. Please make sure you have the following prerequisites installed:
* [git](https://git-scm.com/downloads)
* [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
* [wget](https://eternallybored.org/misc/wget/)

Download the wget executable and copy it into one of your folders in System path to use the wget command from anywhere. Use the `path` command in command line to find the folders. Paste the wget.exe in one of the folders given by `path`. An example folder is `C:\Windows`.

pycocotools is not supported by default on Windows. Use the pycocotools build for Windows at [here](https://github.com/philferriere/cocoapi). Instead of `pip install pycocotools`, using:
```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

Visual C++ Build Tools are required by the build. Everything else is the same.

## Distributed Training

The pose estimation neural network can take days to train.
To speed up training, we support distributed GPU training.
We use the [KungFu](https://github.com/lsds/KungFu) library to scale out training.
KungFu is very easy to install and run (compared to the previously used Horovod library
which depends on OpenMPI), and simply follow
the [instruction](https://github.com/lsds/KungFu#install).

In the following, we assume that you have added `kungfu-run` into the `$PATH`.

(i) To run on a machine with 4 GPUs:

```bash
kungfu-run -np 4 python3 train.py --parallel --kf-optimizer=sma
```

The default KungFu optimizer is `sma` which implements synchronous model averaging.
You can also use other KungFu optimizers: `sync-sgd` (which is the same as the DistributedOptimizer in Horovod)
and `async-sgd` if you train your model in a cluster that has limited bandwidth and straggelers.

(ii) To run on 2 machines (which have the nic `eth0` with IPs as `192.168.0.1` and `192.168.0.2`):

```bash
kungfu-run -np 8 -H 192.168.0.1:4,192.168.0.1:4 -nic eth0 python3 train.py --parallel --kf-optimizer=sma
```

## High-performance Inference using TensorRT

Real-time inference on resource-constrained embedded platforms is always challenging. To resolve this, we provide a TensorRT-compatible inference engine.
The engine has two C++ APIs, both defined in `include/openpose-plus.hpp`.
They are for running the TensorFlow model with TensorRT and post-processing respectively.

For details of inference(dependencies/quick start), please refer to [**cpp-inference**](doc/markdown-doc/cpp-inference.md).

We are improving the performance of the engine.
Initial benchmark results for running the full OpenPose model are as follows.
On Jetson TX2, the inference speed is 13 frames / second (the mobilenet variant is even faster).
On Jetson TX1, the speed is 10 frames / second. On Titan 1050, the speed is 38 frames / second. 

After our first optimization, we achieved 50FPS(float32) on 1070Ti. 

We also have a Python binding for the engine. The current binding relies on
the external tf-pose-estimation project. We are working on providing the Python binding for our high-performance
C++ implementation. For now, to enable the binding, please build C++ library for post processing by:

```bash
./scripts/install-pafprocess.sh
# swig is required. Run `conda install -c anaconda swig` to install swig.
```

See [tf-pose](https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess) for details.

## Live Camera Example

You can look at the examples in the `examples` folder to see how to use the inference C++ APIs.
Running `./scripts/live-camera.sh` will give you a quick review of how it works.

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

## License

You can use the project code under a free [Apache 2.0 license](https://github.com/tensorlayer/tensorlayer/blob/master/LICENSE.rst) ONLY IF you:
- Cite the [TensorLayer paper](https://github.com/tensorlayer/tensorlayer#cite) and this project in your research article if you are an **academic user**.
- Acknowledge TensorLayer and this project in your project websites/articles if you are a **commercial user**.

## Related Discussion

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
