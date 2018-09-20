#!/bin/sh
set -e

export HAVE_CUDA=1

make

D=$HOME/var/data/openpose/126

# IMAGE=$HOME/Downloads/new-tests/cam0_27.png
IMAGES=$D/cam2_3938.png,$D/cam1_2386.png

# MODEL_FILE=$HOME/Downloads/vgg.uff
MODEL_FILE=$HOME/Downloads/vggtiny.uff

./cmake-build/$(uname -s)/uff-runner_main \
    --model_file=${MODEL_FILE} \
    --image_files=${IMAGES}
