#!/bin/sh
set -e

MODEL_DIR=$HOME/Downloads
DATA_DIR=$HOME/var/data/openpose

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

test_vggtiny_model() {
    local images=$(echo $@ | tr ' ' ',')
    echo ${images}
    ./uff-runner.py \
        --base-model=vggtiny \
        --path-to-npz=${MODEL_DIR}/new-models/hao18/pose350000.npz \
        --images=${images}
}

test_hao28_model() {
    local images=$(echo $@ | tr ' ' ',')
    echo ${images}
    ./uff-runner.py \
        --base-model=hao28_experimental \
        --path-to-npz=${MODEL_DIR}/hao28/pose345000.npz \
        --images=${images}
}

# test_vgg_model \
#     ${DATA_DIR}/examples/media/COCO_val2014_000000000192.jpg \
#     ${DATA_DIR}/new-tests/cam0_27.png

# test_vggtiny_model \
#     ${DATA_DIR}/examples/media/COCO_val2014_000000000192.jpg \
#     ${DATA_DIR}/new-tests/cam0_27.png

test_hao28_model \
    ${DATA_DIR}/examples/media/COCO_val2014_000000000192.jpg \
    ${DATA_DIR}/new-tests/cam0_27.png
