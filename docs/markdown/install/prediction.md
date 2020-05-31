# C++ Prediction Library Installation

## Prerequisites

* C++ 17 Compiler. (g++7, clang++4.0, MSVC19.0 or newer)
* CMake 3.5+ (older version may also work but not tested)
* Third-Party
    * OpenCV3+.
    * [CUDA](https://developer.nvidia.com/cuda-downloads), [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt_304/tensorrt-install-guide/index.html).
    * gFlags(optional, for examples/tests)

## Build On Ubuntu 18.04

```bash
# >>> Install OpenCV3.
sudo apt -y install cmake libopencv-dev 
# You may also install OpenCV from source to get best performance.

# >>> Install CUDA/TensorRT

# >>> Build gFlags(Optional) from source. Install it if you want to run the examples.
wget https://github.com/gflags/gflags/archive/v2.2.2.zip
unzip v2.2.2.zip
cd gflags-2.2.2
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON && make
sudo make install

# >>> Build HyperPose
git clone https://github.com/tensorlayer/hyperpose.git
cd hyperpose
mkdir build && cd build
cmake .. && make -j$(nproc)
```

Just go to [Quick Start](../quick_start/prediction.md) to test your installation.