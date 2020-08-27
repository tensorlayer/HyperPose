#!/bin/sh

set -e

[ $(which gdown) ] || (echo "Downloading gdown via PIP" && python3 -m pip install gdown -U)

model_name="ppn-resnet50-V2-HW=384x384.onnx"

BASEDIR=$(realpath "$(dirname $0)")
cd $BASEDIR
mkdir -p ../data/models
cd ../data/models

python3 "$BASEDIR/downloader.py" --model $model_name
