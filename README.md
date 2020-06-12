# HyperPose

[![Documentation Status](https://readthedocs.org/projects/hyperpose/badge/?version=latest)](https://hyperpose.readthedocs.io/en/latest/?badge=latest)

HyperPose is a library for building human pose estimation systems that can efficiently operate in the wild.

> **Note**: We are in the process of migrating our APIs from 1.0 to 2.0. The migration is expected to finish by July 2020.

## Features

HyperPose has two key features, which are not available in existing libraries:

- **Flexible training platform**: HyperPose provides flexible Python APIs to build many useful pose estimation models (e.g., OpenPose and PoseProposalNetwork). HyperPose users can, for example, customize data augmentation, use parallel GPUs for training, and replace deep neural networks (e.g., changing from ResNet to MobileNet), thus building models specific to their real-world scenarios.
- **High-performance pose estimation**: HyperPose achieves real-time pose estimation though a high-performance pose estimation engine. This engine implements numerous system optimizations: pipeline parallelism, model inference with TensorRT, CPU/GPU hybrid scheduling, and many others. This allows HyperPose to run 4x faster than OpenPose and 10x faster than TF-Pose.

## Get Started

You can install HyperPose and learn its APIs through [Documentation](https://hyperpose.readthedocs.io/en/latest/).

## Example

We provide an example to show human pose estimation achieved by HyperPose. You need to install CUDA Toolkit 10+, TensorRT 7+, OpenCV 3.2+ and gFlags (cmake version), and enable C++ 17 support. Once the prerequisite are ready, run the following script:

```bash
sudo apt -y install subversion curl
sh scripts/download-test-data.sh         # Install data for examples.
sh scripts/download-tinyvgg-model.sh     # Install tiny-vgg model.
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make -j$(nproc) # Build library && examples.
./example.operator_api_batched_images_paf    # The ouput images will be in the build folder.
```

## Performance

We compare the prediction performance of HyperPose with [OpenPose 1.6](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [TF-Pose](https://github.com/ildoonet/tf-pose-estimation). We implement the OpenPose algorithms with different configurations in HyperPose. The test-bed has Ubuntu18.04, 1070Ti GPU, Intel i7 CPU (12 logic cores). The test video is Crazy Updown Funk ([YouTube](https://www.youtube.com/watch?v=2DiQUX11YaY)). The HyperPose models (in the ONNX or Uff formats) are available [here](https://github.com/tensorlayer/pretrained-models/tree/master/models/hyperpose).

| HyperPose Configuration  | DNN Size | DNN Input Shape | HyerPose | Baseline |
| --------------- | ------------- | ------------------ | ------------------ | --------------------- |
| OpenPose (VGG)   | 209.3MB       | 656 x 368            | 27.32 FPS           | 8 FPS (OpenPose)          |
| OpenPose (TinyVGG)  | 34.7 MB       | 384 x 256          | 124.925 FPS         | N/A                   |
| OpenPose (MobileNet) | 17.9 MB       | 432 x 368          | 84.32 FPS           | 8.5 FPS (TF-Pose)         |
| OpenPose (ResNet18)  | 45.0 MB       | 432 x 368          | 62.52 FPS           | N/A                  |

As we can see, HyperPose is the only library that can achieve **real-time** human pose estimation.

</a>
<p align="center">
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/dance_foot.gif?raw=true", width="360">
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
