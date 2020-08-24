# Performance of Prediction Library

## Result

| Method                   | Backbone Size | Network Resolution | Operator API / FPS | Stream API / FPS | Other Framework / FPS | Batch Size |
| ------------------------ | ------------- | ------------------ | ------------------ | ---------------- | --------------------- | ---------- |
| OpenPose COCO            | 209.3MB       | 656 x 368          | 19.78              | 27.32            | 8 (OpenPose)          | 8          |
| Tiny VGG + PAF           | 34.7 MB       | 384 x 256          | 66.62              | 124.925          | /                     | 8          |
| MobileNet + PAF          | 17.9 MB       | 432 x 368          | 50.89              | 84.32            | /                     | 8          |
| ResNet50 + PAF           | 45.0 MB       | 432 x 368          | 50.89              | 84.32            | 8.5 (TF-Pose)         | 8          |
| ResNet18 + Pose Proposal | 50.3 MB       | 384 x 384          | 212.42             | 349.17           | /                     | 64         |

> **Environment**: System@Ubuntu18.04, GPU@1070Ti, CPU@i7(12 logic cores). 
>
> **Tested Video Source**: Crazy Updown Funk(resolution@640x360, frame_count@7458, source@[YouTube](https://www.youtube.com/watch?v=2DiQUX11YaY))

> OpenPose performance is not tested with batch processing as it seems not to be implemented. (see [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/issues/100)) 

## Suggestions

- PAF post processing is slow. Batch processing will not accelerate PAF and will bring little improvement in the speed.
- And Pose Proposal post processing is fast(over 8k FPS in single core). So any optimization(e.g. batch processing) in DNN inference will be remarkable for the throughput of the pipeline. For example, using batch size 8 we got 164 FPS, using batch size 64 we got 349 FPS, and using batch size 128 we got 383 FPS.