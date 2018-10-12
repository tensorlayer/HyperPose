#!/bin/bash
set -e

now_nano() {
    date +%s*1000000000+%N | bc
}

measure() {
    echo "[->] $@ begins"
    local begin=$(now_nano)
    "$@"
    local end=$(now_nano)
    local duration=$(echo "scale=6; ($end - $begin) / 1000000000" | bc)
    echo "[==] $@ took ${duration}s" | tee -a time.log
}

cd $(dirname $0)/..

export PYTHONUNBUFFERED=1
# DATA_DIR=$(pwd)/data/media

MODEL_DIR=${HOME}/Downloads
DATA_DIR=$HOME/var/data/openpose

IMAGES=$(ls ${DATA_DIR}/examples/media/*.jpg | sort | tr '\n' ',')

REPEAT=10
LIMIT=1

profile_model() {
    local model=$1
    local npz=$2

    local log_name=$model-$3
    if [ $3 == "NHWC" ]; then
        local data_format=channels_last
    elif [ $3 == "NCHW" ]; then
        local data_format=channels_first
    else
        echo "invalid data format, NHWC or NCHW is requied"
        return
    fi

    ./examples/example-inference-1.py --path-to-npz=${MODEL_DIR}/$npz \
        --base-model=$model \
        --images=${IMAGES} \
        --data-format=$data_format \
        --plot='1' \
        --repeat=${REPEAT} \
        --limit=${LIMIT}
}

mkdir -p logs
measure profile_model vggtiny new-models/hao18/pose350000.npz NHWC
# measure profile_model vggtiny new-models/hao18/pose350000.npz NCHW
# measure profile_model mobilenet mbn280000.npz NHWC
# measure profile_model vgg vgg450000_no_cpm.npz NHWC
# measure profile_model vgg vgg450000_no_cpm.npz NCHW
# measure profile_model hao28_experimental hao28/pose345000.npz NHWC
# measure profile_model hao28_experimental hao28/pose345000.npz NCHW
