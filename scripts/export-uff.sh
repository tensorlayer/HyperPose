#!/bin/sh
set -e

cd $(dirname $0)/..

MODEL_DIR=${HOME}/Downloads

DATA_FORMAT=channels_first # Must use channels_first

./export.py --data-format=${DATA_FORMAT} --base-model=vgg --path-to-npz=${MODEL_DIR}/vgg450000_no_cpm.npz --uff-filename=${MODEL_DIR}/vgg.uff
./export.py --data-format=${DATA_FORMAT} --base-model=vggtiny --path-to-npz=${MODEL_DIR}/new-models/hao18/pose350000.npz --uff-filename=${MODEL_DIR}/vggtiny.uff
./export.py --data-format=${DATA_FORMAT} --base-model=mobilenet --path-to-npz=${MODEL_DIR}/mbn280000.npz --uff-filename=mobilenet.uff
