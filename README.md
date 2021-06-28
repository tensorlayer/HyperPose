</a>
<p align="center">
    <img src="./docs/markdown/images/logo.png", width="600">
</p>

<p align="center">
    <a href="https://readthedocs.org/projects/hyperpose/badge/?version=latest" title="Docs Building"><img src="https://readthedocs.org/projects/hyperpose/badge/?version=latest"></a>
    <a href="https://github.com/tensorlayer/hyperpose/actions?query=workflow%3ACI" title="Build Status"><img src="https://github.com/tensorlayer/hyperpose/workflows/CI/badge.svg"></a>
    <a href="https://hub.docker.com/r/tensorlayer/hyperpose" title="Docker"><img src="https://img.shields.io/docker/image-size/tensorlayer/hyperpose"></a>
    <a href="https://drive.google.com/drive/folders/1w9EjMkrjxOmMw3Rf6fXXkiv_ge7M99jR?usp=sharing" title="PreTrainedModels"><img src="https://img.shields.io/badge/trained%20models-GoogleDrive-brightgreen.svg"></a>
    <a href="https://en.cppreference.com/w/cpp/17" title="CppStandard"><img src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"></a>
    <a href="https://github.com/tensorlayer/hyperpose/graphs/commit-activity" title="Maintenance"><img src="https://img.shields.io/badge/maintained%3F-YES-brightgreen.svg"></a>
    <a href="https://github.com/tensorlayer/tensorlayer/blob/master/LICENSE.rst" title="TensorLayer"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
</p>

---

<p align="center">
    <a href="#Features">Features</a> •
    <a href="#Documentation">Documentation</a> •
    <a href="#Quick-Start-with-Docker">Quick-Start with Docker</a> •
    <a href="#Performance">Performance</a> •
    <a href="#Accuracy">Accuracy</a> •
    <a href="#License">License</a>
</p>

# HyperPose

HyperPose is a library for building high-performance custom pose estimation systems.

## Features

HyperPose has two key features:

- **High-performance pose estimation with parallel CPUs/GPUs**: HyperPose achieves real-time pose estimation through a high-performance pose estimation engine. This engine implements numerous system optimisations: pipeline parallelism, model inference with TensorRT, CPU/GPU hybrid scheduling, and many others. These optimisations contribute to up to 10x higher FPS compared to OpenPose and TF-Pose.
- **Flexibility for developing custom pose estimation models**: HyperPose provides high-level Python APIs to develop pose estimation models. HyperPose users can:
    * Customise training, evaluation, visualisation, pre-processing and post-processing in pose estimation models (e.g., OpenPose, Pifpaf, PoseProposal Network).
    * Customise model architectures and training datasets.
    * Seamlessly scale-out training to multiple GPUs.

## Quick Start

The HyperPose library contains two parts:
* A C++ library for high-performance pose estimation model inference.
* A Python library for developing custom pose estimation models.

### C++ inference library

The easiest way to use the inference library is through a [Docker image](https://hub.docker.com/r/tensorlayer/hyperpose). Pre-requisites for this image:

* [CUDA Driver](https://www.tensorflow.org/install/gpu) (>= 418.81.07)
* [NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) (>= 2.0)
* [Docker CE Engine](https://docs.docker.com/engine/install/) (>= 19.03)

Run this script to check is pre-requisites are ready:

```bash
wget https://raw.githubusercontent.com/tensorlayer/hyperpose/master/scripts/test_docker.py -qO- | python
```

Once pre-requisites are installed, pull
the HyperPose docker:

```bash
docker pull tensorlayer/hyperpose
```

We provide 4 examples within this image (The following commands have been tested with Ubuntu 18.04):

```bash
# [Example 1]: Doing inference on given video, copy the output.avi to the local path.
docker run --name quick-start --gpus all tensorlayer/hyperpose --runtime=stream
docker cp quick-start:/hyperpose/build/output.avi .
docker rm quick-start


# [Example 2](X11 server required to see the imshow window): Real-time inference.
# You may need to install X11 server locally:
# sudo apt install xorg openbox xauth
xhost +; docker run --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix tensorlayer/hyperpose --imshow


# [Example 3]: Camera + imshow window
xhost +; docker run --name pose-camera --rm --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/video0:/dev/video0 tensorlayer/hyperpose --source=camera --imshow
# To quit this image, please type `docker kill pose-camera` in another terminal.


# [Dive into the image]
xhost +; docker run --rm --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/video0:/dev/video0 --entrypoint /bin/bash tensorlayer/hyperpose
# For users that cannot access a camera or X11 server. You may also use:
# docker run --rm --gpus all -it --entrypoint /bin/bash tensorlayer/hyperpose
```

More information of the Docker image is [here](https://hyperpose.readthedocs.io/en/latest/markdown/quick_start/prediction.html#table-of-flags-for-hyperpose-cli).

### Python training library

We recommend to use the Python training library within an [Anaconda](https://www.anaconda.com/products/individual) environment. Test environments:<br>
* Ubuntu 18.04, Tesla V100-DGX, NVIDIA Driver 440.33.01, CUDA 10.2
* Ubuntu 18.04, Tesla V100-DGX, NVIDIA Driver 410.79, CUDA 10.0
* Ubuntu 18.04, TITAN RTX, NVIDIA Driver 430.64, CUDA 10.1
* Ubuntu 18.04, TITAN XP, NVIDIA Driver 430.26, CUDA 10.2
* Ubuntu 16.04, RTX 2080Ti, NVIDIA Driver 430.50, CUDA 10.1

Once Anaconda is installed, run below Bash commands to create a virtual environment:

```bash
# Create virtual environment (choose yes)
conda create -n hyperpose python=3.7
# Activate the virtual environment, start installation
conda activate hyperpose
# Install cudatoolkit and cudnn library using conda
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.0
```

We then install the dependencies listed in [requirements.txt](https://github.com/tensorlayer/hyperpose/blob/master/requirements.txt):

```bash
pip install -r requirements.txt
```

We show how to train a custom pose estimation model with HyperPose. HyperPose APIs contain three key modules: *Config*, *Model* and *Dataset*, and their basic usages are shown below.

```python
import tensorflow as tf
import tensorlayer as tl
tf.test.is_gpu_available()

from hyperpose import Config, Model, Dataset
# Set model name to distinguish models (necessary)
Config.set_model_name("My_lopps")
# Set model type, model backbone and dataset
Config.set_model_type(Config.MODEL.LightweightOpenpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
Config.set_dataset_type(Config.DATA.MSCOCO)
# Set single-node training or parallel-training
Config.set_train_type(Config.TRAIN.Single_train)
config = Config.get_config()
model = Model.get_model(config)
dataset = Dataset.get_dataset(config)
train = Model.get_train(config)
# Start the training process
train(model,dataset)
```

The full training program is [here](https://github.com/tensorlayer/hyperpose/blob/master/train.py). To evaluate the trained model, you can use an evaluation program [here](https://github.com/tensorlayer/hyperpose/blob/master/eval.py). More information about the training library is [here](https://hyperpose.readthedocs.io/en/latest/markdown/quick_start/training.html).


## Documentation

The APIs of the HyperPose training library and the inference library are described in [Documentation](https://hyperpose.readthedocs.io/en/latest/).

## Performance

We compare the prediction performance of HyperPose with [OpenPose 1.6](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [TF-Pose](https://github.com/ildoonet/tf-pose-estimation). We implement the OpenPose algorithms with different configurations in HyperPose. The test-bed has Ubuntu18.04, 1070Ti GPU, Intel i7 CPU (12 logic cores).

| HyperPose Configuration  | DNN Size | Input Size | HyperPose | Baseline |
| --------------- | ------------- | ------------------ | ------------------ | --------------------- |
| OpenPose (VGG)   | 209.3MB       | 656 x 368            | **27.32 FPS**           | 8 FPS (OpenPose)          |
| OpenPose (TinyVGG)  | 34.7 MB       | 384 x 256          | **124.925 FPS**         | N/A                   |
| OpenPose (MobileNet) | 17.9 MB       | 432 x 368          | **84.32 FPS**           | 8.5 FPS (TF-Pose)         |
| OpenPose (ResNet18)  | 45.0 MB       | 432 x 368          | **62.52 FPS**           | N/A                  |
| OpenPifPaf (ResNet50)  | 97.6 MB       | 432 x 368          | **44.16 FPS**           | 14.5 FPS (OpenPifPaf)    |

## Accuracy

We evaluate the accuracy of pose estimation models developed by HyperPose. The environment is Ubuntu16.04, with 4 V100-DGXs and 24 Intel Xeon CPU. The training procedure takes 1~2 weeks using 1 V100-DGX for each model. (If you don't want to train from scratch, you could use our pre-trained backbone models)

| HyperPose Configuration | DNN Size | Input Size | Evaluate Dataset | Accuracy-hyperpose (Iou=0.50:0.95) | Accuracy-original (Iou=0.50:0.95) |
| -------------------- | ---------- | ------------- | ---------------- | --------------------- | ----------------------- |
| OpenPose (VGG19)   | 199 MB | 432 x 368 | MSCOCO2014 (random 1160 images) | 57.0 map | 58.4 map  |
| LightweightOpenPose (Dilated MobileNet)   | 17.7 MB | 432 x 368 | MSCOCO2017(all 5000 img.) | 46.1 map | 42.8 map |
| LightweightOpenPose (MobileNet-Thin)   | 17.4 MB | 432 x 368 | MSCOCO2017 (all 5000 img.) | 44.2 map | 28.06 map (MSCOCO2014) |
| LightweightOpenPose (tiny VGG)   | 23.6 MB | 432 x 368 | MSCOCO2017 (all 5000 img.) | 47.3 map | - |
| LightweightOpenPose (ResNet50)   | 42.7 MB | 432 x 368 | MSCOCO2017 (all 5000 img.) | 48.2 map | - |
| PoseProposal (ResNet18)   | 45.2 MB | 384 x 384 | MPII (all 2729 img.) | 54.9 map (PCKh) | 72.8 map (PCKh)|


</a>
<p align="center">
    <img src="./docs/markdown/images/demo-xbd.gif", width="600">
</p>

<p align="center">
    新宝岛 with HyperPose(Lightweight OpenPose model)
</p>

## License

HyperPose is open-sourced under the [Apache 2.0 license](https://github.com/tensorlayer/tensorlayer/blob/master/LICENSE.rst).

<!-- - Please acknowledge TensorLayer and this project in your project websites/articles if you are a **commercial user**. -->

<!-- ## Related Discussion

- [TensorLayer Slack](https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc)
- [TensorLayer WeChat](https://github.com/tensorlayer/tensorlayer-chinese/blob/master/docs/wechat_group.md)
- [TensorLayer Issues 434](https://github.com/tensorlayer/tensorlayer/issues/434)
- [TensorLayer Issues 416](https://github.com/tensorlayer/tensorlayer/issues/416) -->

<!--

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is the state-of-the-art hyperpose estimation algorithm.
In its Caffe [codebase](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation),
data augmentation, training, and neural networks are most hard-coded. They are difficult
to be customized. In addition,
key performance features such as embedded platform supports and parallel GPU training are missing.
All these limitations makes OpenPose, in these days, hard to
be deployed in the wild. To resolve this, we develop **OpenPose-Plus**, a high-performance yet flexible hyperpose estimation framework that offers many powerful features:

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

The hyperpose estimation neural network can take days to train.
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
the external tf-hyperpose-estimation project. We are working on providing the Python binding for our high-performance
C++ implementation. For now, to enable the binding, please build C++ library for post processing by:

```bash
./scripts/install-pafprocess.sh
# swig is required. Run `conda install -c anaconda swig` to install swig.
```

See [tf-hyperpose](https://github.com/ildoonet/tf-hyperpose-estimation/tree/master/tf_pose/pafprocess) for details.

## Live Camera Example

You can look at the examples in the `examples` folder to see how to use the inference C++ APIs.
Running `./scripts/live-camera.sh` will give you a quick review of how it works.

## License

You can use the project code under a free [Apache 2.0 license](https://github.com/tensorlayer/tensorlayer/blob/master/LICENSE.rst) ONLY IF you:
- Cite the [TensorLayer paper](https://github.com/tensorlayer/tensorlayer#cite) and this project in your research article if you are an **academic user**.
- Acknowledge TensorLayer and this project in your project websites/articles if you are a **commercial user**.

## Related Discussion

- [TensorLayer Slack](https://join.slack.com/t/tensorlayer/shared_invite/enQtMjUyMjczMzU2Njg4LWI0MWU0MDFkOWY2YjQ4YjVhMzI5M2VlZmE4YTNhNGY1NjZhMzUwMmQ2MTc0YWRjMjQzMjdjMTg2MWQ2ZWJhYzc)
- [TensorLayer WeChat](https://github.com/tensorlayer/tensorlayer-chinese/blob/master/docs/wechat_group.md)
- [TensorLayer Issues 434](https://github.com/tensorlayer/tensorlayer/issues/434)
- [TensorLayer Issues 416](https://github.com/tensorlayer/tensorlayer/issues/416)

-->

<!--

## Paper's Model

- [Default MPII](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_MPI/pose_deploy.prototxt)
- [Default COCO model](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation/blob/master/model/_trained_COCO/pose_deploy.prototxt)
- [Visualizing Caffe model](http://ethereon.github.io/netscope/#/editor)
-->
