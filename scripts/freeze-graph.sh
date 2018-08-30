#!/bin/sh
set -e

cd $(dirname $0)/..
./export.py

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

python3 ${FREEZE_GRAPH_BIN} \
    --input_graph ${ROOT}/${GRAPH_FILE} \
    --input_checkpoint ${ROOT}/${CHECKPOINT} \
    --output_graph ${ROOT}/${OUTPUT_GRAPH} \
    --output_node_names ${OUTPUT_NODE_NAMES}
