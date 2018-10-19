#!/bin/sh
set -e

make
echo

MODEL_DIR=$HOME/Downloads
MODEL_FILE=${MODEL_DIR}/hao28-600000-256x384.uff

MODEL_URL=https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/openpose-plus/hao28-600000-256x384.uff

if [ ! -f ${MODEL_FILE} ]; then
    echo "${MODEL_FILE} NOT exist."
    echo "Please generate it from trained npz, see export-uff.sh for more details."
    echo "Or download demo model from ${MODEL_URL}"
    exit 1
fi

gksize=9

run() {
    local BIN=$(pwd)/cmake-build/$(uname -s)/example-live-camera

    local buffer_size=4

    DISPLAY=:0 \
        ${BIN} \
        --input_height=256 \
        --input_width=384 \
        --buffer_size=${buffer_size} \
        --use_f16 \
        --gauss_kernel_size=${gksize} \
        --model_file=${MODEL_FILE}
}

run
