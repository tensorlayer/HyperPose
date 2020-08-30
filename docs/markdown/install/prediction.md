# C++ Prediction Library Installation

## Docker Environment Installation

To ease the installation, you can use HyperPose library in our docker image where the environment is pre-installed.

### Official Docker Image

NVidia docker support is required to execute our docker image. 

The official image is on [DockerHub](https://hub.docker.com/r/tensorlayer/hyperpose).

```bash
# Pull the latest image.
docker pull tensorlayer/hyperpose

# Dive into the image. (Connect local camera and imshow window)
xhost +; docker run --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/video0:/dev/video0 --entrypoint /bin/bash tensorlayer/hyperpose
# For users that cannot access a camera or X11 server. You may also use:
# docker run --rm --gpus all -it --entrypoint /bin/bash tensorlayer/hyperpose
```

Note that the entry point is the [`hyperpose-cli`](https://hyperpose.readthedocs.io/en/latest/markdown/quick_start/prediction.html#table-of-flags-for-hyperpose-cli) binary in the build directory.

### Build docker from source

```bash
# Enter the repository folder.
USER_DEF_NAME=my_hyperpose
docker build -t $(USER_DEF_NAME) .
docker run --gpus all $(USER_DEF_NAME)
```

## Build From Source

### Prerequisites

* C++ 17 Compiler. (g++7, clang++4.0, MSVC19.0 or newer)
* CMake 3.5+ 
* Third-Party
    * OpenCV3.2+. (**[OpenCV 4+](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html) is highly recommended**)
    * [CUDA 10.2](https://developer.nvidia.com/cuda-downloads), [CuDNN 7.6.5](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html), [TensorRT 7.0](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
    * gFlags(for command-line tool/examples/tests)

> Other versions of the packages may also work but not tested.

> **About TensorRT installation**
>
> - For Linux users, you highly recommended to install it in a system-wide setting. You can install TensorRT7 via the [debian distributions](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian) or [NVIDIA network repo](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#maclearn-net-repo-install)(CUDA and CuDNN dependency will be automatically installed).
> - Different TensorRT version requires specific CUDA and CuDNN version. For specific CUDA and CuDNN requirements of TensorRT7, please refer to [this](https://docs.nvidia.com/deeplearning/tensorrt/support-matrix/index.html#platform-matrix).
> - Also, for Ubuntu 18.04 users, this [3rd party blog](https://ddkang.github.io/2020/01/02/installing-tensorrt.html) may help you. 

### Build On Ubuntu 18.04

```bash
# >>> Install OpenCV3+ and other dependencies. 
sudo apt -y install cmake libopencv-dev libgflags-dev
# !Note that the APT version OpenCV3.2 on Ubuntu18.04 has some trouble on Cameras Newer version is suggested.
# You are highly recommended to install OpenCV 4+ from scratch also for better performance.

# >>> Install dependencies to run the scripts in `${REPO}/scripts`
sudo apt install python3-dev python3-pip 

# >>> Install CUDA/CuDNN/TensorRT: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-debian

# >>> Build HyperPose
git clone https://github.com/tensorlayer/hyperpose.git
cd hyperpose
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build .
```

## Build with User Codes

You can directly write codes and execute it under the hyperpose repository.

- **Step 1**: Write your own codes in `hyperpose/examples/user_codes` with suffix `.cpp`.
- **Step 2**:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_USER_CODES=ON   # BUILD_USER_CODES is by default on
cmake --build .
```

- **Step 3**: Execute your codes!

Just go to [Quick Start](../quick_start/prediction.md) to test your installation.