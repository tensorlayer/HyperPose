# OpenPose-Plus: Real-time and Flexible Pose Estimation

</a>
<p align="center">
    <img src="https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/media/dance_foot.gif?raw=true", width="360">
</p>

[![Documentation Status](https://readthedocs.org/projects/openpose-plus/badge/?version=latest)](https://openpose-plus.readthedocs.io/en/latest/?badge=latest)

[OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) is the state-of-the-art pose estimation algorithm.
In its Caffe [codebase](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation),
data augmentation, training, and neural networks are most hard-coded. They are difficult
to be customised for achieving the best performance in real-world applications.
Necessary features such as embedded platform supports and parallel GPU training are missing as well.
This motivates us to develop OpenPose-Plus, a real-time and flexible pose estimation framework that offers many powerful features:
- Flexible combination of standard training dataset with your own custom labelled data.
- Customisable data augmentation pipeline without compromising performance
- Deployment on embedded platforms using TensorRT
- Switchable neural networks (e.g., changing VGG to MobileNet for minimal memory consumption)
- High-performance training using multiple GPUs

This project is under active development, some TODOs are as follows:
- Parallel training (experimental support)
- Pose Proposal Networks, ECCV 2018

## Custom Model Training

Training the model is implemented using TensorFlow. To run `train.py`, you would need to install packages, shown
in [requirements.txt](https://github.com/tensorlayer/openpose-plus/blob/master/requirements.txt), in your virtual environment (Python <=3.6):

```bash
pip install -r requirements.txt
pip install pycocotools
```

`train.py` automatically download MSCOCO 2017 dataset into `dataset/coco17`.
The default model is VGG19 used in the OpenPose paper.
To customize the model, simply changing it in `models.py`.

You can use `config.py` to configure the training. `config.DATA.train_data` can be:
* `coco`: training data is COCO dataset only (default)
* `custom`: training data is your dataset specified by `config.DATA.your_xxx`
* `coco_and_custom`: training data is COCO and your dataset

`config.MODEL.name` can be:
* `vgg`: VGG19 version (default), slow
* `vggtiny`: VGG tiny version, faster
* `mobilenet`: MobileNet version, faster

Train your model by running:

```bash
python train.py
```

## Training using Multiple GPUs

The pose estimation neural network can take days to train.
To speed up the training, we support multiple GPU training while requiring
minimal changes in your code. We use Horovod to support training on GPUs that can spread across multiple machines. 
You need to install the [OpenMPI](https://www.open-mpi.org/) in your machine.
We also provide an example script (`scripts/install-mpi.sh`) to help you go through the installation. 
Once OpenMPI is installed, you can install Horovod python library as follows:

```bash
pip install horovod
```

To enable paralle training, set the `config.TRAIN.train_mode` to `parallel` (default is `single`).

(i) To run on a machine with 4 GPUs:

```bash
$ mpirun -np 4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```

(ii) To run on 4 machines with 4 GPUs each:

```bash
$ mpirun -np 16 \
    -H server1:4,server2:4,server3:4,server4:4 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python train.py
```


## High-performance Inference using TensorRT

Real-time inference on resource-constrained embedded platforms
is important but challenging. To resolve this, we provide a TensorRT-compatible inference engine. 
The engine has two C++ APIs, both defined in `include/openpose-plus.hpp`.
They are for running the TensorFlow model with TensorRT and post-processing respectively.

You can look at the examples in the `examples` folder to see how to use the APIs.
Running `./scripts/live-camera.sh` will give you a quick review of how it works.

You can build the APIs into a standard C++ library by just running `make pack`, provided that you have the following dependencies installed

  - tensorRT
  - opencv
  - gflags

We are improving the performance of the engine. 
Initial benchmark results are as follows.
On Jetson TX 2, the inference speed is 13 frames / second. On Jetson TX1, the 
speed is 10 frames / second. On Titan 1050, the 
speed is 38 frames / second.

We also have a Python binding for the engine. The current binding relies on
the external tf-pose-estimation project. We are working on providing the Python binding for our high-performance
C++ implementation. For now, to enable the binding, please build C++ library for post processing by:

```bash
cd inference/pafprocess
make

# ** before recompiling **
rm -rf build
rm *.so
```

See [tf-pose](https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess) for details.

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
- Cite the [TensorLayer paper](https://github.com/tensorlayer/tensorlayer#cite) and this project in your research article if as an **academic user**.
- Acknowledge TensorLayer and this project in your project websites/articles as a **commercial user**.

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
