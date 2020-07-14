# Python Training Library Installation

## Prerequisites
* [anaconda3](https://www.anaconda.com/products/individual)
* [CUDA](https://developer.nvidia.com/cuda-downloads)

## Configure environment
Hyperpose training library can be directly used by putting Hyperpose in the directory and import.
But it has to install the prerequist environment to make it available.

The following instructions have been tested on the environments below:
* Ubuntu 18.04, Tesla V100-DGXStation, Nvidia Driver Version 440.33.01, CUDA Verison=10.2  
* Ubuntu 18.04, Tesla V100-DGXStation, Nvidia Driver Version 410.79, CUDA Verison=10.0  
* Ubuntu 18.04, TITAN RTX, Nvidia Driver Version 430.64, CUDA Version=10.1  
* Ubuntu 18.04, TITAN Xp, Nvidia Driver Version 430.26, CUDA Version=10.2

```bash
# >>> create virtual environment (choose yes)
conda create -n hyperpose python=3.7
# >>> activate the virtual environment, start installation
conda activate hyperpose
# >>> install cuda and cudnn using conda
conda install cudatoolkit=10.0.130
conda install cudnn=7.6.0
# >>> install tensorflow of version 2.0.0
pip install tensorflow-gpu==2.0.0
# >>> install the newest version tensorlayer from github
pip install tensorlayer==2.2.3
# >>> install other requirements (numpy<=17.0.0 because it has conflicts with pycocotools)
pip install opencv-python
pip install numpy==1.16.4
pip install pycocotools
pip install matplotlib
# >>> now the configuration is done, check whether the GPU is avaliable.
python
>>> import tensorflow as tf
>>> import tensorlayer as tl
>>> tf.test.is_gpu_available()
# >>> if the output is true, congratulation! you can import and run hyperpose now
>>> from hyperpose import Config,Model,Dataset
```
## Extra configuration for exporting model
For training, the above configuration is enough, but to export model into **onnx** format for inference,one should install the
following two extra library:
* tf2onnx (necessary ,used to convert .pb format model into .onnx format model) [reference](https://github.com/onnx/tensorflow-onnx)
```bash
pip install -U tf2onnx
```
* graph_transforms (unnecesary,used to check the input and output node of the .pb file if one doesn't know)
build graph_transforms according to [reference](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#using-the-graph-transform-tool)




