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


### Using system-wise CUDA toolkits

Users may also directly depend on the system-wise CUDA and CuDNN libraries.

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

To check your system-wise CuDNN version **on Linux**: the output (in the comment) shows that we have CuDNN 8.0.5.
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
pip install hyperpose
```

Or you can install a specific release of hyperpose from GitHub, for example:

```bash
export HYPERPOSE_VERSION="2.2.0-alpha"
pip install https://github.com/tensorlayer/hyperpose/archive/${HYPERPOSE_VERSION}.zip
```

More GitHub releases and its version can be found [here](https://github.com/tensorlayer/hyperpose/releases).

### Local installation

You can also install HyperPose by installing the raw GitHub repository, this is usually for developers.

```bash
# Install the source codes from GitHub
git clone https://github.com/tensorlayer/hyperpose.git
pip install -r hyperpose/requirements.txt

# Add `hyperpose/hyperpose` to `PYTHONPATH` to help python find it.
export HYPERPOSE_PYTHON_HOME=$(pwd)/hyperpose
export PYTHONPATH=$HYPERPOSE_PYTHON_HOME/python:${PYTHONPATH}
```

## Check the installation

Let's check whether HyperPose is successfully installed by running following commands:

```bash
python -c '
import tensorflow as tf             # Test TensorFlow installation
import tensorlayer as tl            # Test TensorLayer installation
assert tf.test.is_gpu_available()   # Test GPU existence
import hyperpose                    # Test HyperPose import
'
```

## Optional Setup

### Extra configuration for model exportation

The hypeprose python training library handles the whole pipelines for developing the pose estimation system, including training, evaluating and testing. Its goal is to produce a **.npz** file that contains the well-trained model weights.

For the training platform, the enviroment configuration above is engough. However, most inference engine accepts `.pb` or [`.onnx`](https://onnx.ai/) format model. For example, the HyperPose C++ inference engine leverages [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) as the DNN engine, which takes `.onnx` models as inputs.

Thus, one need to convert the trained model loaded with **.npz** file weight to **.pb** format or **.onnx** format for further deployment, which need extra configuration below:

#### Converting a `.pb` model

To convert the model into `.pb` format, we use `@tf.function` to decorate the `infer` function for each model class, and we then can use the `get_concrete_function` function from tensorflow to consctruct the frozen model computation graph and then save it with `.pb` format.

We provide [a commandline tool](https://github.com/tensorlayer/hyperpose/blob/master/export_pb.py) to facilitate the conversion. The prerequisite of this tool is a tensorflow library installed along with HyperPose's dependency.

#### Converting a `.onnx` model

To convert a trained model into `.onnx` format, we need to first convert the model into `.pb` format, we then convert a `.pb` model into `.onnx` format, which requires 2 additional libraries:

* [**tf2onnx**](https://github.com/onnx/tensorflow-onnx) for converting TensorFlow's `.pb` model into `.onnx` format.
* [**graph_transforms**](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#using-the-graph-transform-tool) 

To install `tf2onnx`, we simply run:

```bash
pip install -U tf2onnx
```

After converting a `.pb` file to an `.onnx` file using tf2onnx, it is usually required to provide the input node name and output node name of the computation graph stored in `.pb` file, which is often tedious. Instead, we use `graph_transform` to finding out the input and output node of the `.pb` model file automatically. 

build graph_transforms according to [tensorflow tools](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#using-the-graph-transform-tool).

### Extra configuration for distributed training with KungFu

The HyperPose python training library can also perform distributed training with [Kungfu](https://github.com/lsds/KungFu). To enable parallel training, please install [Kungfu](https://github.com/lsds/KungFu) according to its official instructon.
