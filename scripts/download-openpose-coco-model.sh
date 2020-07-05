#!/bin/sh

set -e

model_name="openpose_coco.onnx"
model_md5="9f422740c7d41d93d6fe16408b0274ef"

cd $(dirname $0)
mkdir -p ../data/models
cd ../data/models

if [ ! -f "$model_name" -o "$(md5sum "$model_name" | cut -d ' ' -f 1)" != "$model_md5" ] ; then
    echo "Installing $model_name ..."
    URL="https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/hyperpose/$model_name"
    curl -vLOJ $URL
fi

echo "$model_name installed!"
