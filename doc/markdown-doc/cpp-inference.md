# OpenPose-Plus: Inference

### Dependencies

To build high-performance inference using OpenPose-Plus, the following dependencies should be installed on your machine:

- OpenCV3.4+
- GFlags
- TensorRT(version 5 & 7 are tested, other versions may also work)

#### OpenCV 3.4+

- [Linux](https://docs.opencv.org/trunk/d7/d9f/tutorial_linux_install.html)

#### GFlags

- Ubuntu:

```shell
sudo apt-get install libgflags-dev
```

#### TensorRT

Please follow the instructions [here](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar).

Make sure that the header files & libraries of TensorRT & CUDA are in the `/usr/local/cuda/targets/${ARCH}-linux`'s `include` & `lib` folder.

Or you may need to set `CUDA_RT` as `/path/to/TensorRT` when executing cmake commands.

### Quick Start

```shell
# Tools we need when building.
sudo apt-get install subversion 

make pack # If you want to enable profiling tracing, use `make pack_trace`
./scripts/download-test-data.sh
./scripts/download-pretrained-inf-models.sh
cd cmake-build/Linux
./example-batch-detector 
# Or `./example-batch-detector --use_f16=true` if your GPU support float16
# Use `./example-batch-detector --help` to find out the usage of flags.  
```

Then you'll see the output images in the `cmake-build/Linux` folder.

If you cannot execute the binary file successfully, you can refer to the `tensorrt.log` for more details.

### Speed Results

If you want to share your results on your machine, welcome to PR!

#### 1070Ti + Intel@i7(12 cores)

- FP32: ~50 FPS