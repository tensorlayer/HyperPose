# Frequently Asked Questions

## Installation

### No C++17 Compiler(Linux)?

* **Using `apt` as your package manager?**
    * Install from `ppa`.
    * Helpful link: [LINK](https://gist.github.com/jlblancoc/99521194aba975286c80f93e47966dc5).
* **Otherwise**
    * Build a C++17 compiler from source.

### Build without examples/tests?

```cmake
cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF
```

### Build OpenCV from source?

Refer to [here](https://www.learnopencv.com/tag/install/).

### Network problem when installing the test models/data from the command line?

Download them manually:

- All prediction models are available [here](https://github.com/tensorlayer/pretrained-models/tree/master/models/hyperpose).
- The test data are taken from the [OpenPose Project](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/media).

## Training 

## Prediction

### Performance?

- Usually the 1st try of execution(cold start) on small amount of data tends to be slow. 
You can use a longer video/more images to test the performance(or run it more than once).
- The performance is mainly related to the followings(you can customize the followings):
    - **The complexity of model**(not only FLOPS but also parameter numbers): smaller is usually better.
    - **The model network resolution**(alse see [here](../performance/prediction.md)): smaller is better.
    - **The input / output size**(this mainly involves in the speed of `cv::resize`): smaller is better.
    - **The upsampling factor of the feature map when doing post processing**: smaller is better. 
    (By default the PAF parser will upsample the feature map by 4x. We did this according to the [Lightweight-OpenPose](https://arxiv.org/abs/1811.12004) paper.)
- Use better hardware(Good CPUs can make the post-processing faster!).
- Use SIMD instructions of your CPU. (Compile OpenCV from source and enable the instruction sets in cmake configuration)