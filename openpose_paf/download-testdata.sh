#!/bin/sh
set -e

cd $(dirname $0)
if [ ! -f network-outputs ]; then
    time curl -sLOJ https://github.com/tensorlayer/fast-openpose/files/2378505/network-outputs.gz
    gzip -d network-outputs.gz
fi
tar -xf network-outputs
