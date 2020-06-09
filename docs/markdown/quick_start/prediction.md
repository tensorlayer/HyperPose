# Quick Start of Prediction Library

## Prerequisites

* Make sure you have HyperPose installed. (if not, you can refer to [here](../install/prediction.md)).
* Make sure you have `svn`(subversion) and `curl` installed. (will be used in command line scripts)

For Linux users, you may:

```bash
sudo apt -y install subversion curl
```

## Install Test Data

```bash
# cd to the git repo.

sh scripts/download-test-data.sh
```

> You download them manually to `${HyperPose}/data/media/` via [LINK](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/media) **if the network is not working**.

## Install Test Models

```bash
# cd to the git repo. And download pre-trained models you want. 

sh scripts/download-openpose-thin-model.sh      # ~20  MB
sh scripts/download-tinyvgg-model.sh            # ~30  MB
sh scripts/download-openpose-res50-model.sh     # ~45  MB
sh scripts/download-openpose-coco-model.sh      # ~200 MB
```

> You can download them manually to `${HyperPose}/data/models/` via [LINK](https://github.com/tensorlayer/pretrained-models/tree/master/models/hyperpose) **if the network is not working**.

## Predict a sequence of images

> To see the flags of the following examples, just type `${exe} --help`

### Using a fast model

```bash
# cd to your build directory.

# Take images in ../data/media as a big batch and do prediction.

./example.operator_api_batched_images_paf
# The same as: `./example.operator_api_batched_images_paf --model_file ../data/models/hao28-600000-256x384.uff --input_folder ../data/media --input_width 384 --input_height 256`
```

The output images will be in the build folder.

### Using a precise model

```bash
./example.operator_api_batched_images_paf --model_file ../data/models/openpose_thin.onnx --input_width 432 --input_height 368 

./example.operator_api_batched_images_paf --model_file ../data/models/openpose_coco.onnx --input_width 656 --input_height 368 
```

### Convert models into TensorRT Engine Protobuf format

You may find that it takes one or two minutes before the real prediction starts. This is because TensorRT will try to profile the model to get a optimized runtime model. 

To save the model conversion time, you can convert it in advance.

```bash
./example.gen_serialized_engine --model_file ../data/models/openpose_coco.onnx --input_width 656 --input_height 368 --max_batch_size 20
# You'll get ../data/models/openpose_coco.onnx.trt

# Use the converted model to do prediction
./example.operator_api_batched_images_paf --model_file ../data/models/openpose_coco.onnx.trt --input_width 656 --input_height 368
```

## Predict a video using Operator API

```bash
./example.operator_api_video_paf  # Using the fast model by default.
```

The output video will be in the building folder.

## Predict a video using Stream API(faster)

```bash
./example.stream_api_video_paf    # Using the fast model by default.
```

## Play with camera

```camera
./example.operator_api_imshow_paf --camera
```