#!/bin/sh
set -e

HAVE_CUDA=1 make
echo

MODEL_DIR=$HOME/Downloads
D=$HOME/var/data/openpose

batch_size=4
repeat=20
gksize=13

# BIN=$(pwd)/cmake-build/$(uname -s)/example
BIN=$(pwd)/cmake-build/$(uname -s)/example-stream-detector

run_uff_cpp() {
    local MODEL_FILE=${MODEL_DIR}/hao28-256x384.uff
    local IMAGES=$(echo $@ | tr ' ' ',')
    ${BIN} \
        --input_height=256 \
        --input_width=384 \
        --batch_size=${batch_size} \
        --use_f16 \
        --gauss_kernel_size=${gksize} \
        --repeat ${repeat} \
        --model_file=${MODEL_FILE} \
        --image_files=${IMAGES}
}

run_uff_cpp \
    $D/examples/media/COCO_val2014_000000000192.png \
    $D/new-tests/cam0_27.png \
    $D/126/cam2_3938.png \
    $D/126/cam1_2386.png
