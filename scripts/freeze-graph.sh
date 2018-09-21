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

cd $(dirname $0)/..
ROOT=$(pwd)

TF_VERSION=$(python3 -c "import tensorflow as tf; print(tf.__version__)")
TF_TAG=v${TF_VERSION}

FREEZE_GRAPH_URL=https://raw.githubusercontent.com/tensorflow/tensorflow/${TF_TAG}/tensorflow/python/tools/freeze_graph.py
FREEZE_GRAPH_BIN=${ROOT}/scripts/freeze_graph.py

[ ! -f ${FREEZE_GRAPH_BIN} ] && curl -s ${FREEZE_GRAPH_URL} >${FREEZE_GRAPH_BIN}

CHECKPOINT_DIR=$(pwd)/checkpoints

GRAPH_FILE=${CHECKPOINT_DIR}/graph.pb.txt
CHECKPOINT=${CHECKPOINT_DIR}/saved_checkpoint-0
OUTPUT_GRAPH=${CHECKPOINT_DIR}/freezed

OUTPUT_NODE_NAMES=image,outputs/conf,outputs/paf

freeze() {
    python3 ${FREEZE_GRAPH_BIN} \
        --input_graph ${GRAPH_FILE} \
        --input_checkpoint ${CHECKPOINT} \
        --output_graph ${OUTPUT_GRAPH} \
        --output_node_names ${OUTPUT_NODE_NAMES}
}

BASE_MODEL=vgg
PATH_TO_NPZ=${HOME}/Downloads/vgg450000_no_cpm.npz

# BASE_MODEL=mobilenet
# PATH_TO_NPZ=${HOME}/Downloads/mbn28000.npz

measure ./export.py --base-model=${BASE_MODEL} --path-to-npz=${PATH_TO_NPZ} --graph-filename=graph.pb.txt
measure freeze
