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

GRAPH_FILE=checkpoints/graph.pb.txt
CHECKPOINT=checkpoints/saved_checkpoint-0
OUTPUT_GRAPH=checkpoints/freezed

OUTPUT_NODE_NAMES=image,upsample_size,upsample_heatmat,tensor_peaks,upsample_pafmat

freeze() {
    python3 ${FREEZE_GRAPH_BIN} \
        --input_graph ${ROOT}/${GRAPH_FILE} \
        --input_checkpoint ${ROOT}/${CHECKPOINT} \
        --output_graph ${ROOT}/${OUTPUT_GRAPH} \
        --output_node_names ${OUTPUT_NODE_NAMES}
}

measure ./inference/export.py
measure freeze
