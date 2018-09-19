#!/bin/sh
set -e

MODEL_DIR=$HOME/Downloads
DATA_DIR=$HOME/Downloads/new-tests

# cam0_27.png
# cam0_59.png
# cam2_21.png
# cam3_107.png
# cam3_146.png
# cam3_148.png
# cam3_52.png
# cam3_63.png

test_vgg_model() {
    local images=$(echo $@ | tr ' ' ',')
    echo ${images}
    ./uff-runner.py \
        --base-model=vgg \
        --path-to-npz=${MODEL_DIR}/vgg450000_no_cpm.npz \
        --images=${images}
}

test_hao28_model() {
    local images=$(echo $@ | tr ' ' ',')
    echo ${images}
    ./uff-runner.py \
        --base-model=hao28 \
        --path-to-npz=models/pose345000.npz \
        --images=${images}
}

test_vgg_model \
    ./data/media/COCO_val2014_000000000192.jpg \
    ${DATA_DIR}/cam0_27.png

# test_hao28_model \
#     ./data/media/COCO_val2014_000000000192.jpg \
#     ${DATA_DIR}/cam0_27.png
