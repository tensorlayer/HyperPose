#!/bin/sh

set -e

[ $(which gdown) ] || (echo "Downloading gdown via PIP" && python3 -m pip install gdown -U)

model_name="TinyVGG-V1-HW=256x384.uff"

BASEDIR=$(realpath "$(dirname $0)")
cd $BASEDIR
mkdir -p ../data/models
cd ../data/models

python3 "$BASEDIR/downloader.py" --model $model_name
