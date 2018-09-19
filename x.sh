#!/bin/sh
set -e

export HAVE_CUDA=1

make

D=$HOME/Desktop/126

# IMAGE=$HOME/Downloads/new-tests/cam0_27.png
IMAGES=$D/cam2_3938.png,$D/cam1_2386.png

./cmake-build/$(uname -s)/uff-runner_main \
    --model_file=$HOME/Downloads/vgg.uff \
    --image_files=${IMAGES}
