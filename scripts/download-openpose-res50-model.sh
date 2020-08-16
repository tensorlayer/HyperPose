#!/bin/sh

set -e

[ $(which gdown) ] || (echo "Downloading gdown via PIP" && python3 -m pip install gdown)

model_name="lopps-resnet50-V2-HW=368x432.onnx"
model_md5="38c0ad11c76d23f438e1bd1a32101409"
gdrive_file_id="1tb8jnXkoiscfr-ZVydAALg7dtUwAKdEd"

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