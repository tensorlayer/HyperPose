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

TF_ROOT=${ROOT}/cpp/tensorflow

FREEZE_GRAPH_BIN=${TF_ROOT}/tensorflow/python/tools/freeze_graph.py

GRAPH_FILE=checkpoints/graph.pb.txt
CHECKPOINT=checkpoints/saved_checkpoint-0
OUTPUT_GRAPH=checkpoints/freezed

name1=model/cpm/stage6/branch1/conf/BiasAdd
name2=model/cpm/stage6/branch2/pafs/BiasAdd
name3=Select # the peek tensor
OUTPUT_NODE_NAMES=${name1},${name2},${name3}

freeze() {
    python3 ${FREEZE_GRAPH_BIN} \
        --input_graph ${ROOT}/${GRAPH_FILE} \
        --input_checkpoint ${ROOT}/${CHECKPOINT} \
        --output_graph ${ROOT}/${OUTPUT_GRAPH} \
        --output_node_names ${OUTPUT_NODE_NAMES}
}

measure ./export.py
measure freeze
