#!/bin/sh

set -e

model_name="hao28-600000-256x384.uff"
model_md5="6551931d16e55cc9370c5c13d91383c3"

cd $(dirname $0)
mkdir -p ../data/models
cd ../data/models

if [ ! -f "$model_name" -o "$(md5sum "$model_name" | cut -d ' ' -f 1)" != "$model_md5" ] ; then
    echo "Installing $model_name ..."
    URL="https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/hyperpose/$model_name"
    curl -vLOJ $URL
fi

echo "$model_name installed!"
