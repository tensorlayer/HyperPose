#!/bin/sh

set -e

cd openpose_paf

if [ ! -f network-outputs ]; then
    time curl -sLOJ https://github.com/tensorlayer/fast-openpose/files/2378505/network-outputs.gz
    gzip -d network-outputs.gz
fi
tar -xf network-outputs

make
./cmake-build/$(uname -s)/test_paf >stdout.log 2>stderr.log
code trace.log
