#!/bin/sh

set -e

[ $(which gdown) ] || echo "Downloading gdown via PIP" || python3 -m pip install gdown

model_name="ppn-resnet50-V2-HW=384x384.onnx"
model_md5="0d1df2e61c0f550185d562ec67a5f2ca"
gdrive_file_id="1qMSipZ5_QMyRuNQ7ux5isNxwr678ctwG"

cd $(dirname $0)
mkdir -p ../data/models
cd ../data/models

if [ ! -f "$model_name" ] || [ "$(md5sum "$model_name" | cut -d ' ' -f 1)" != "$model_md5" ] ; then
    echo "Installing $model_name ..."
    gdown "https://drive.google.com/uc?id=$gdrive_file_id"
fi

if [ "$(md5sum "$model_name" | cut -d ' ' -f 1)" != "$model_md5" ] ; then
    echo "Failed to install $model_name. The MD5 code doesn't match!"
else
    echo "$model_name installed!"
fi