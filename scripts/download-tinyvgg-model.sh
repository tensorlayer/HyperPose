#!/bin/sh

set -e

[ $(which gdown) ] || (echo "Downloading gdown via PIP" && python3 -m pip install gdown)

model_name="TinyVGG-V1-HW=256x384.uff"
model_md5="6551931d16e55cc9370c5c13d91383c3"
gdrive_file_id="1KlKjNMaruJnNYEXQKqzHGqECBAmwB92T"

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
