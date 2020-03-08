#!/bin/sh

set -e

cd $(dirname $0)
if [ ! -d ../data/models ]; then
    mkdir -p ../data/models
fi
cd ../data/models

echo "Installing Pretrained VGG uff model file..."
if [ ! -f hao28-600000-256x384.uff ]; then
    URL=https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/openpose-plus/hao28-600000-256x384.uff
    curl -vLOJ $URL
fi
echo "Installation completed!"
