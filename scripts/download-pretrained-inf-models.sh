#!/bin/sh

set -e

cd $(dirname $0)
[ ! -d ../data/models ] && mkdir ../data/models
cd ../data/models

echo "Installing Pretrained VGG uff model file..."
[ ! -e hao28-600000-256x384.uff ]  && svn export https://github.com/tensorlayer/pretrained-models/trunk/models/openpose-plus/hao28-600000-256x384.uff
echo "Installation completed!"