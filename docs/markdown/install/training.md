# Python Training Library Installation

## Configure CUDA environment

You can configure your CUDA either by Anaconda or your system setting.

### Using CUDA toolkits from Anaconda (RECOMMENDED)

:::{admonition} Prerequisites
- [Anaconda3](https://www.anaconda.com/products/individual)
- [NVidia Driver >= 410.79 (required by CUDA 10)](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#driver-installation)
:::

It is suggested to create new conda environment regarding the CUDA requirements.

```bash
# >>> create virtual environment
conda create -n hyperpose python=3.7 -y
# >>> activate the virtual environment, start installation
conda activate hyperpose
# >>> install cudatoolkit and cudnn library using conda
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.0
```
 
::::{warning}
It is also possible to install CUDA dependencies without creating a new environment. 
But it might introduce environment conflicts.

:::{code-block} bash
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.0
:::
::::


### Using system-wide CUDA toolkits

Users may also directly depend on the system-wide CUDA and CuDNN libraries.

HyperPose have been tested on the environments below:

| OS           | NVIDIA Driver | CUDA Toolkit | GPU            |
| ------------ | ------------- | ------------ | -------------- |
| Ubuntu 18.04 | 410.79        | 10.0         | Tesla V100-DGX |
| Ubuntu 18.04 | 440.33.01     | 10.2         | Tesla V100-DGX |
| Ubuntu 18.04 | 430.64        | 10.1         | TITAN RTX      |
| Ubuntu 18.04 | 430.26        | 10.2         | TITAN XP       |
| Ubuntu 16.04 | 430.50        | 10.1         | RTX 2080Ti     |

::::{admonition} Check CUDA/CuDNN versions

To test CUDA version, run `nvcc --version`: the highlight line in the output indicates that you have CUDA 11.2 installed.
:::{code-block} bash
:emphasize-lines: 5
nvcc --version
# ========== Valid output looks like ==========
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2020 NVIDIA Corporation
# Built on Mon_Nov_30_19:08:53_PST_2020
# Cuda compilation tools, release 11.2, V11.2.67
# Build cuda_11.2.r11.2/compiler.29373293_0
:::

To check your system-wide CuDNN version **on Linux**: the output (in the comment) shows that we have CuDNN 8.0.5.
:::{code-block} bash
ls /usr/local/cuda/lib64 | grep libcudnn.so
# === Valid output looks like ===
# libcudnn.so
# libcudnn.so.8
# libcudnn.so.8.0.5
:::
::::

## Install HyperPose Python training library

### Install with `pip`

To install a stable library from [Python Package Index](https://github.com/tensorlayer/hyperpose):

```bash
pip install -U hyperpose
```

Or you can install a specific release of hyperpose from GitHub, for example:

```bash
export HYPERPOSE_VERSION="2.2.0-alpha"
pip install -U https://github.com/tensorlayer/hyperpose/archive/${HYPERPOSE_VERSION}.zip
```

More GitHub releases and its version can be found [here](https://github.com/tensorlayer/hyperpose/releases).

### Local installation

You can also install HyperPose by installing the raw GitHub repository, this is usually for developers.

```bash
# Install the source codes from GitHub
git clone https://github.com/tensorlayer/hyperpose.git
pip install -U -r hyperpose/requirements.txt

# Add `hyperpose/hyperpose` to `PYTHONPATH` to help python find it.
export HYPERPOSE_PYTHON_HOME=$(pwd)/hyperpose
export PYTHONPATH=$HYPERPOSE_PYTHON_HOME/python:${PYTHONPATH}
```

## Check the installation

Let's check whether HyperPose is installed by running following commands:

```bash
python -c '
import tensorflow as tf             # Test TensorFlow installation
import tensorlayer as tl            # Test TensorLayer installation
assert tf.test.is_gpu_available()   # Test GPU availability
import hyperpose                    # Test HyperPose import
'
```

## Optional Setup

### Extra configurations for exporting models

The hypeprose python training library handles the whole pipelines for developing the pose estimation system, including training, evaluating and testing. Its goal is to produce a **.npz** file that contains the well-trained model weights.

For the training platform, the enviroment configuration above is engough. However, most inference engine accepts ProtoBuf or [ONNX](https://onnx.ai/) format model. For example, the HyperPose C++ inference engine leverages [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) as the DNN engine, which takes ONNX models as inputs.

Thus, one need to convert the trained model loaded with **.npz** file weight to **.pb** format or **.onnx** format for further deployment, which need extra configuration below:

#### Converting a ProtoBuf model

To convert the model into ProtoBuf format, we use `@tf.function` to decorate the `infer` function for each model class, and we then can use the `get_concrete_function` function from tensorflow to consctruct the frozen model computation graph and then save it with ProtoBuf format.

We provide [a commandline tool](https://github.com/tensorlayer/hyperpose/blob/master/export_pb.py) to facilitate the conversion. The prerequisite of this tool is a tensorflow library installed along with HyperPose's dependency.

#### Converting a ONNX model

To convert a trained model into ONNX format, we need to first convert the model into ProtoBuf format, we then convert a ProtoBuf model into ONNX format, which requires an additional library: [**tf2onnx**](https://github.com/onnx/tensorflow-onnx) for converting TensorFlow's ProtoBuf model into ONNX format.

To install `tf2onnx`, we simply run:

```bash
pip install -U tf2onnx
```

### Extra configuration for distributed training with KungFu

The HyperPose python training library can also perform distributed training with [Kungfu](https://github.com/lsds/KungFu). To enable parallel training, please install [Kungfu](https://github.com/lsds/KungFu) according to its official instructon.
