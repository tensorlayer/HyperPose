#!/bin/sh

set -e

model_name="hao28-600000-256x384.uff"

cd $(dirname $0)
if [ ! -d ../data/models ]; then
    mkdir -p ../data/models
fi
cd ../data/models

echo "Installing Pretrained VGG uff model file..."
if [ ! -f "$model_name" ]; then
    curl -LOJ https://github.com/tensorlayer/pretrained-models/trunk/models/openpose-plus/hao28-600000-256x384.uff
fi

mini_size=1048576
if [ $(wc -c < "$model_name") -le $mini_size ]; then
    warning_text="[Warning!] File size is less than $mini_size! Re-downloading..."
    echo "\033[33m$warning_text\033[0m"
    rm $model_name
    wget --no-check-certificate https://media.githubusercontent.com/media/tensorlayer/pretrained-models/master/models/openpose-plus/hao28-600000-256x384.uff
fi

echo "Installation completed!"
