# Quick Start of Prediction Library

::::{admonition} Prerequisites
1. Have HyperPose Inference Library installed ([HowTo](../install/prediction.md)).
2. Make sure `python3` and `python3-pip` are installed.

For Linux user, we can simply install them with `apt`.
:::{code-block} bash
sudo apt -y install subversion python3 python3-pip
:::
::::

:::{warning}
HyperPose Inference Library is mostly compatible and tested under Linux, especially Ubuntu 18.04. Please try Ubuntu 18.04 or a docker container for best experience.
:::

## Data preparation

### Test data

We install a folder called `media/` and put it in `${HyperPose_HOME}/data/media/`.

```bash
# cd to the git repo.
sh scripts/download-test-data.sh
```

:::{admonition} Manual installation
:class: important
If you have trouble installing the test data through command line, you can manually download the data folder from [LINK](https://github.com/CMU-Perceptual-Computing-Lab/openpose/tree/master/examples/media), and put it in `${HyperPose_HOME}/data/media/`.
:::

### Install test models

The following scripts will download pre-trained under `${HyperPose_HOME}/data/models/`.

```bash
# cd to the git repo. And download pre-trained models you want. 

sh scripts/download-openpose-thin-model.sh      # ~20  MB
sh scripts/download-tinyvgg-model.sh            # ~30  MB (UFF model)
sh scripts/download-openpose-res50-model.sh     # ~45  MB
sh scripts/download-openpose-coco-model.sh      # ~200 MB
sh scripts/download-openpose-mobile-model.sh
sh scripts/download-tinyvgg-v2-model.sh
sh scripts/download-openpose-mobile-model.sh
sh scripts/download-openpifpaf-model.sh         # ~98  MB (OpenPifPaf)
sh scripts/download-ppn-res50-model.sh          # ~50  MB (PoseProposal)
```


:::{admonition} Manual installation
:class: tip
You can manually install them from our Model Zoo at [GoogleDrive](https://drive.google.com/drive/folders/1w9EjMkrjxOmMw3Rf6fXXkiv_ge7M99jR?usp=sharing).
:::

## Predict a sequence of images

::::{admonition} Note for docker users
:class: note
The following tutorial commands are based on HyperPose commandline tool `hyperpose-cli`, which is also the entry point of the container (locates in `/hyperpose/build/hyperpose-cli`). If you are playing with a container, please first get into the container in interactive mode.
:::{code-block} bash
# Without imshow/camera functionality
docker run --rm --gpus all -it tensorlayer/hyperpose
# With imshow/camera functionality
xhost +; docker run --rm --gpus all -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix --device=/dev/video0:/dev/video0 --entrypoint /bin/bash tensorlayer/hyperpose

# Once get inside the image
cd /hyperpose/build
:::
::::

### Using a fast model

```bash
# cd to your build directory.

# Predict all images in `../data/media`
./hyperpose-cli --source ../data/media --model ../data/models/lopps-resnet50-V2-HW=368x432.onnx --w 368 --h 432
# The source flag can be ignored as the default value is `../data/media`.
```

The output images will dumped into the build folder by default. For more models, please look at `/hyperpose/data/models`. Their name indicates their input tensor shape.

:::{admonition} Ignore error message from `uff` models.
:class: caution
If you are using the `TinyVGG-V1-HW=256x384.uff` (`.uff` models are going to be deprecated by TensorRT), you may meet logging messages like `ERROR: Tensor image cannot be both input and output`. This is harmless and please just ignore it. 
:::

### Table of flags for `hyperpose-cli`

Note that the entry point of our official docker image is also `hyperpose-cli` in the `/hyperpose/build` folder.

| Flag           | Meaning                                                      | Default                                   |
| -------------- | ------------------------------------------------------------ | ----------------------------------------- |
| `model`          | Path to your model.                                          | ../data/models/TinyVGG-V2-HW=342x368.onnx |
| `source`         | Path to your source. <br />The source can be a folder path (automatically glob all images), a video path, an image path or the key word `camera` to open your camera. | ../data/media/video.avi                   |
| `post`           | Post-processing methods. This key can be `paf` or `ppn`.     | paf                                       |
| `keep_ratio`     | The DNN takes a fixed input size, where the images must resize to fit that input resolution. However, not hurt the original human scale, we may want to resize by padding. And this is flag enable you to do inference without break original human ratio. (Good for accuracy) | true                                      |
| `w`              | The input width of your model. Currently, the trained models we provided all have specific requirements for input resolution. | 432 (for the tiny-vgg model)         |
| `h`              | The input height of your model.                              | 368 (for the tiny-vgg model)         |
| `max_batch_size` | Maximum batch size for inference engine to execute.          | 8                                         |
| `runtime`        | Which runtime type to use. This can be `operator` or `stream`. If you want to open your camera or producing `imshow` window, please use `operator`. For better processing throughput on videos, please use `stream`. | operator                                  |
| `imshow`         | true                                                         | Whether to open an `imshow` window.       |
| `saving_prefix`  | The output media resource will be named after `$(saving_prefix)_$(ID).$(format)` | "output"                                  |
| `alpha`          | The weight of key point visualization. (from 0 to 1)         | 0.5   
| `logging`        | Print the internal logging information or not.               | false                                    |

:::{seealso}
Run `./hyperpose-cli --help` for the usage.
:::

### Using OpenPose-based (PAF) models

```bash
./hyperpose-cli --model ../data/models/openpose-thin-V2-HW=368x432.onnx --w 432 --h 368

./hyperpose-cli --model ../data/models/openpose-coco-V2-HW=368x656.onnx --w 656 --h 368
```

### Use PifPaf model

Set `--post` flag to `pifpaf` to enable a [PifPaf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kreiss_PifPaf_Composite_Fields_for_Human_Pose_Estimation_CVPR_2019_paper.pdf) model processing.

```bash
./hyperpose-cli --model ../data/models/openpifpaf-resnet50-HW=368x432.onnx --w 368 --h 432 --post pifpaf
```

### Convert models into TensorRT Engine Protobuf format

You may find that it takes minutes before the prediction really starts. This is because TensorRT will try to profile the model to get a optimized runtime model. 

You can pre-compile it in advance, to save the model conversion time.

```bash
./example.gen_serialized_engine --model_file ../data/models/openpose-coco-V2-HW=368x656.onnx --input_width 656 --input_height 368 --max_batch_size 16
# You'll get ../data/models/openpose-coco-V2-HW=368x656.onnx.trt
# If you only want to do inference on single images(batch size = 1), please use `--max_batch_size 1` and this will improve the engine's performance.

# Use the converted model to do prediction
./hyperpose-cli --model ../data/models/openpose-coco-V2-HW=368x656.onnx.trt --w 656 --h 368
```

:::{caution}
Currently, we support models in TensorRT `float32` mode. 
Other data types (e.g., `int8`) are not supported at this point (welcome to contribute!).
:::

## Predict a video using Operator API

```bash
./hyperpose-cli --runtime=operator --source=../data/media/video.avi
```

The output video will be in the build folder.

## Predict a video using Stream API (higher throughput)

```bash
./hyperpose-cli --runtime=stream --source=../data/media/video.avi
# In stream API, the imshow functionality will be closed.
```

## Play with the camera

```bash
./hyperpose-cli --source=camera
# Note that camera mode is not compatible with Stream API. If you want to do inference on your camera in real time, the Operator API is designed for it.
```
