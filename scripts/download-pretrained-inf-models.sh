#!/bin/sh

set -e

cd $(dirname $0)
if [ ! -d ../data/models ]; then
    mkdir -p ../data/models
fi
cd ../data/models

echo "Installing Pretrained VGG uff model file..."
if [ ! -f hao28-600000-256x384.uff ]; then
    curl -LOJ https://github.com/tensorlayer/pretrained-models/trunk/models/openpose-plus/hao28-600000-256x384.uff
fi
echo "Installation completed!"
