# Overview

## Why HyperPose

HyperPose provides:

- **Flexible training**
  - Well abstracted APIs(Python) to help you manage the pose estimation pipeline quickly and directly
  - Dataset(COCO, MPII)
  - Pose Estimation Methods
    - Backbones: ResNet, VGG(Tiny/Normal/Thin), Pose Proposal Network.
  - Post-Processing:  Part Association Field(PAF), Pose Proposal Networks.

- **Fast Prediction**
  - Rich operator APIs for you to do fast DNN inference and post-processing.
  - 2 API styles:
    - Operator API(Imperative): HyperPose provides basic operators to do DNN inference and post processing.
    - Stream API(Declarative): HyperPose provides a streaming processing runtime scheduler where users only need to specify the engine, post-processing methods and input/output streams.
  - Model format supports:
    - Uff.
    - ONNX.
    - Cuda Engine Protobuf.
  - Good performance. (see [here](../performance/prediction.md))

## Training Library Design

## Prediction Library Design