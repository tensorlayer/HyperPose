#!/bin/sh
set -e

export HAVE_CUDA=1

make
# otool -L cmake-build/Darwin/tf-runner_main
# ./cmake-build/$(uname -s)/tf-runner_main
./cmake-build/$(uname -s)/uff-runner_main \
    --model_file=$HOME/Downloads/vgg.uff \
    --image_file=$HOME/Downloads/new-tests/cam0_27.png
