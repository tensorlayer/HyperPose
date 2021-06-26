#!/bin/sh

set -e

[ "$(command -v gdown)" ] || (echo "Downloading gdown via PIP" && python3 -m pip install gdown -U)

model_name="openpifpaf-resnet50-HW=368x432.onnx"

BASEDIR=$(realpath "$(dirname "$0")")
cd "$BASEDIR"
mkdir -p ../data/models
cd ../data/models

python3 "$BASEDIR/downloader.py" --model $model_name
