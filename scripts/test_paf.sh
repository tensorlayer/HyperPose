#!/bin/sh

set -e

make

if [ ! -f network-outputs ]; then
    curl -sLOJ https://github.com/tensorlayer/fast-openpose/files/2378505/network-outputs.gz
    gzip -d network-outputs.gz
fi
tar -xf network-outputs

./cmake-build/$(uname -s)/test_paf >stdout.log 2>stderr.log
code trace.log
