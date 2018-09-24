#!/bin/sh
set -e

export HAVE_CUDA=1

make

MODEL_DIR=$HOME/Downloads
D=$HOME/var/data/openpose

run_uff_cpp() {
    local MODEL_FILE=${MODEL_DIR}/hao28.uff
    local IMAGES=$(echo $@ | tr ' ' ',')
    ./cmake-build/$(uname -s)/uff-runner_main \
        --model_file=${MODEL_FILE} \
        --image_files=${IMAGES}
}

run_uff_cpp \
    $D/examples/media/COCO_val2014_000000000192.png \
    $D/new-tests/cam0_27.png \
    $D/126/cam2_3938.png \
    $D/126/cam1_2386.png
