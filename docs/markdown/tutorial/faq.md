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

- All prediction models are available on [Google Drive](https://drive.google.com/drive/folders/1w9EjMkrjxOmMw3Rf6fXXkiv_ge7M99jR?usp=sharing).
- The test data are taken from the [OpenPose Project](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/media).

## Training 

## Prediction

### TensorRT Error?

- See the `tensorrt.log`. (it contains more informations about logging and is located in where you execute the binary)
- You may meet `ERROR: Tensor image cannot be both input and output` when using the `TinyVGG-V1-HW=256x384.uff` model. And just ignore it.

### Performance?

- Usually the 1st try of execution(cold start) on small amount of data tends to be slow. 
You can use a longer video/more images to test the performance(or run it more than once).
- The performance is mainly related to the followings(you can customize the followings):
    - **The complexity of model**(not only FLOPS but also parameter numbers): smaller is usually better.
    - **The model network resolution**(alse see [here](../performance/prediction.md)): smaller is better.
    - **Batch size**: bigger is faster(higher throughput). (For details, you can refer to *Shen*'s [dissertation](https://digital.lib.washington.edu/researchworks/bitstream/handle/1773/43657/Shen_washington_0250E_19617.pdf?sequence=1&isAllowed=y)) 
    - **The input / output size**(this mainly involves in the speed of `cv::resize`): smaller is better.
    - **The upsampling factor of the feature map when doing post processing**: smaller is better. 
    (By default the PAF parser will upsample the feature map by 4x. We did this according to the [Lightweight-OpenPose](https://arxiv.org/abs/1811.12004) paper.)
- Use better hardware(Good CPUs can make the post-processing faster!).
- Use SIMD instructions of your CPU. (Compile OpenCV from source and enable the instruction sets in cmake configuration)