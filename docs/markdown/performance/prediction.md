# Performance of Prediction Library

## Result


We compare the prediction performance of HyperPose with [OpenPose 1.6](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and [TF-Pose](https://github.com/ildoonet/tf-pose-estimation). 
We implement the OpenPose algorithms with different configurations in HyperPose. 

The test-bed has Ubuntu18.04, 1070Ti GPU, Intel i7 CPU (12 logic cores). 

| HyperPose Configuration  | DNN Size | Input Size | HyperPose | Baseline |
| --------------- | ------------- | ------------------ | ------------------ | --------------------- |
| OpenPose (VGG)   | 209.3MB       | 656 x 368            | **27.32 FPS**           | 8 FPS (OpenPose)          |
| OpenPose (TinyVGG)  | 34.7 MB       | 384 x 256          | **124.925 FPS**         | N/A                   |
| OpenPose (MobileNet) | 17.9 MB       | 432 x 368          | **84.32 FPS**           | 8.5 FPS (TF-Pose)         |
| OpenPose (ResNet18)  | 45.0 MB       | 432 x 368          | **62.52 FPS**           | N/A                  |
| OpenPifPaf (ResNet50)  | 97.6 MB       | 97 x 129          | **178.6 FPS**           | 35.3                  |

> **Environment**: System@Ubuntu18.04, GPU@1070Ti, CPU@i7(12 logic cores). 
>
> **Tested Video Source**: Crazy Updown Funk(resolution@640x360, frame_count@7458, source@[YouTube](https://www.youtube.com/watch?v=2DiQUX11YaY))

> OpenPose performance is not tested with batch processing as it seems not to be implemented. (see [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/100)) 
