# Python Training Library Installation

## Prerequisites
* [Anaconda3](https://www.anaconda.com/products/individual):<br>
    Anaconda is used to create virtual environment that facilitates building the running environment and ease the complexity of library depedencies. Here we mainly use it to create virtual python environment and install cuda run-time libraries.
* [CUDA](https://developer.nvidia.com/cuda-downloads):<br>
    CUDA enviroment is essential to run deep learning neural networks on GPUs. The CUDA installation packages to download should match your system and your NVIDIA Driver version. 

## Configure environment
There are two ways to install hyperpose python training library.

All the following instructions have been tested on the environments below:<br>
> Ubuntu 18.04, Tesla V100-DGXStation, Nvidia Driver Version 440.33.01, CUDA Verison=10.2  
> Ubuntu 18.04, Tesla V100-DGXStation, Nvidia Driver Version 410.79, CUDA Verison=10.0  
> Ubuntu 18.04, TITAN RTX, Nvidia Driver Version 430.64, CUDA Version=10.1  
> Ubuntu 18.04, TITAN Xp, Nvidia Driver Version 430.26, CUDA Version=10.2

Before all, we recommend you to create anaconda virtual environment first, which could handle the possible conflicts between the libraries you already have in your computers and the libraries hyperpose need to install, and also handle the dependencies of the cudatoolkit and cudnn library in a very simple way.<br>
To create the virtual environment, run the following command in bash:
```bash
# >>> create virtual environment (choose yes)
conda create -n hyperpose python=3.7
# >>> activate the virtual environment, start installation
conda activate hyperpose
# >>> install cudatoolkit and cudnn library using conda
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.0
```

After configuring and activating conda enviroment, we can then begin to install the hyperpose.<br>

(I)The first method to install is to put hyperpose python module in the working directory.(recommand)<br>
After git-cloning the source [repository](https://github.com/tensorlayer/hyperpose.git), you can directly import hyperpose python library under the root directory of the cloned repository.<br>

To make importion available, you should install the prerequist dependencies as followed:<br>
you can either install according to the requirements.txt in the [repository](https://github.com/tensorlayer/hyperpose.git)
```bash
    # install according to the requirements.txt
    pip install -r requirements.txt
```

or install libraries one by one

```bash
    # >>> install tensorflow of version 2.3.1
    pip install tensorflow-gpu==2.3.1
    # >>> install tensorlayer of version 2.2.3
    pip install tensorlayer==2.2.3
    # >>> install other requirements (numpy<=17.0.0 because it has conflicts with pycocotools)
    pip install opencv-python
    pip install numpy==1.16.4
    pip install pycocotools
    pip install matplotlib
``` 

This method of installation use the latest source code and thus is less likely to meet compatibility problems.<br><br>

(II)The second method to install is to use pypi repositories.<br>
We have already upload hyperpose python library to pypi website so you can install it using pip, which gives you the last stable version.

```bash
    pip install hyperpose
```

This will download and install all dependencies automatically.

Now after installing dependent libraries and hyperpose itself, let's check whether the installation successes.
run following command in bash:
```bash
# >>> now the configuration is done, check whether the GPU is avaliable.
python
>>> import tensorflow as tf
>>> import tensorlayer as tl
>>> tf.test.is_gpu_available()
# >>> if the output is True, congratulation! you can import and run hyperpose now
>>> from hyperpose import Config,Model,Dataset
```

## Extra configuration for exporting model
The hypeprose python training library handles the whole pipelines for developing the pose estimation system, including training, evaluating and testing. Its goal is to produce a **.npz** file that contains the well-trained model weights.

For the training platform, the enviroment configuration above is engough. However, most inference engine only accept .pb format or .onnx format model, such as [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

Thus, one need to convert the trained model loaded with **.npz** file weight to **.pb** format or **.onnx** format for further deployment, which need extra configuration below:<br>

> **(I)Convert to .pb format:**<br>
    To convert the model into .pb format, we use *@tf.function* to decorate the *infer* function of each model class, so we can use the *get_concrete_function* function from tensorflow to consctruct the frozen model computation graph and then save it in .pb format.

    We already provide a script with cli to facilitate conversion, which located at [export_pb.py](https://github.com/tensorlayer/hyperpose/blob/master/export_pb.py). What we need here is only *tensorflow* library that we already installed.

> **(II)Convert to .onnx format:**<br>
    To convert the model in .onnx format, we need to first convert the model into .pb format, then convert it from .pb format into .onnx format. Two extra library are needed:
> **tf2onnx**:<br>
    *tf2onnx* is used to convert .pb format model into .onnx format model, is necessary here. details information see [reference](https://github.com/onnx/tensorflow-onnx).
    install tf2onnx by running:
    ```bash
    pip install -U tf2onnx
    ```

> **graph_transforms**:<br>
    *graph_transform* is used to check the input and output node of the .pb file if one doesn't know. when convert .pb file into .onnx file using tf2onnx, one is required to provide the input node name and output node name of the computation graph stored in .pb file, so he may need to use *graph_transform* to inspect the .pb file to get node names.<br>
    build graph_transforms according to [tensorflow tools](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#using-the-graph-transform-tool)

## Extra configuration for parallel training
The hyperpose python training library use the High performance distributed machine learning framework **Kungfu** for parallel training.<br>
Thus to use the parallel training functionality of hyperpose, please install [Kungfu](https://github.com/lsds/KungFu) according to the official instructon it provides.



