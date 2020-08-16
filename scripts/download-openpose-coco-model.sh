#!/bin/sh

set -e

[ $(which gdown) ] || (echo "Downloading gdown via PIP" && python3 -m pip install gdown)

model_name="openpose-coco-V2-HW=368x656.onnx"
model_md5="9f422740c7d41d93d6fe16408b0274ef"
gdrive_file_id="15A0SQyPlU2W-Btcf6Ngi6DY0_1CY50d7"

cd $(dirname $0)
mkdir -p ../data/models
cd ../data/models

if [ ! -f "$model_name" ] || [ "$(md5sum "$model_name" | cut -d ' ' -f 1)" != "$model_md5" ] ; then
    echo "Installing $model_name ..."
    python3 -c "import gdown ; gdown.download('"https://drive.google.com/uc?id=$gdrive_file_id"', quiet=False)"
fi

if [ "$(md5sum "$model_name" | cut -d ' ' -f 1)" != "$model_md5" ] ; then
    echo "Failed to install $model_name. The MD5 code doesn't match!"
else
    echo "$model_name installed!"
fi