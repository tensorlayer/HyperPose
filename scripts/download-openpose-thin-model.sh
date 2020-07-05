#!/bin/sh

set -e

model_name="openpose_thin.onnx"
model_md5="65e26d62fd71dc0047c4c319fa3d9096"

cd $(dirname $0)
mkdir -p ../data/models
cd ../data/models

if [ ! -f "$model_name" -o "$(md5sum "$model_name" | cut -d ' ' -f 1)" != "$model_md5" ] ; then
    echo "Installing $model_name ..."
    URL="https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/hyperpose/$model_name"
    curl -vLOJ $URL
fi

echo "$model_name installed!"
