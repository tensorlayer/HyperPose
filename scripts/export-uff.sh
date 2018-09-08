#!/bin/sh
set -e


MODEL_DIR=${HOME}/Downloads

./export.py --base-model=vgg --path-to-npz=${MODEL_DIR}/vgg450000_no_cpm.npz --uff-filename=vgg.uff
./export.py --base-model=vggtiny --path-to-npz=${MODEL_DIR}/pose195000.npz --uff-filename=vggtiny.uff
./export.py --base-model=mobilenet --path-to-npz=${MODEL_DIR}/mbn280000.npz --uff-filename=mobilenet.uff
