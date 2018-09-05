#!/bin/sh
set -e

now_nano() {
    date +%s*1000000000+%N | bc
}

measure() {
    echo "[->] $@ begins"
    local begin=$(now_nano)
    $@
    local end=$(now_nano)
    local duration=$(echo "scale=6; ($end - $begin) / 1000000000" | bc)
    echo "[==] $@ took ${duration}s" | tee -a time.log
}

cd $(dirname $0)
SCRIPT_DIR=$(pwd)

cd ${SCRIPT_DIR}/../../..
ROOT=$(pwd)

batch_limit=200
measure ${SCRIPT_DIR}/cmake-build/$(uname -s)/bin/see-pose \
    --graph_path=checkpoints/freezed \
    --input_images=$(ls data/media/*.jpg | sort | head -n $batch_limit | tr '\n' ',')
