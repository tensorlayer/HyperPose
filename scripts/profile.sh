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
DATA_DIR=${HOME}/var/data/openpose/examples/media

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

    ./test_inference.py --path-to-npz=$HOME/Downloads/$npz \
        --base-model=$model \
        --images=$(ls ${DATA_DIR}/*.jpg | sort | tr '\n' ',') \
        --data-format=$data_format \
        --plot='' \
        --repeat 1 \
        --limit 2

    # >logs/$log_name.stdout.log 2>logs/$log_name.stderr.log
}

mkdir -p logs
# measure profile_model vggtiny pose195000.npz NHWC
# measure profile_model mobilenet mbn280000.npz NHWC
measure profile_model vgg vgg450000_no_cpm.npz NHWC
# measure profile_model vgg vgg450000_no_cpm.npz NCHW # npz shape, is the same, but inference doesn't work yet
