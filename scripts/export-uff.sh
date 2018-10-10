#!/bin/sh
set -e

cd $(dirname $0)/..

MODEL_DIR=${HOME}/Downloads

DATA_FORMAT=channels_first # Must use channels_first

height=256
width=384

export_uff() {
    local base_model=$1
    local npz_file=$2
    local uff_file=$3

    ./export.py \
        --data-format=${DATA_FORMAT} \
        --base-model=${base_model} \
        --path-to-npz=${MODEL_DIR}/${npz_file} \
        --height=${height} \
        --width=${width} \
        --uff-filename=${MODEL_DIR}/${uff_file}

    echo "saved to ${MODEL_DIR}/${uff_file}"
}

# export_uff vgg vgg450000_no_cpm.npz vgg.uff
# export_uff vggtiny new-models/hao18/pose350000.npz vggtiny.uff
export_uff hao28_experimental hao28/pose345000.npz hao28-${height}x${width}.uff
export_uff hao28_experimental pose600000.npz hao28-600000-${height}x${width}.uff

# TODO: make mobilenet support NCHW
# export_uff mobilenet mbn280000.npz mobilenet.uff
