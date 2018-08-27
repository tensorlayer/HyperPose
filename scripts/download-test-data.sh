#!/bin/sh

set -e

cd $(dirname $0)/..
mkdir -p data

# download data/test.jpeg
curl https://github.com/CMU-Perceptual-Computing-Lab/openpose/raw/master/examples/media/COCO_val2014_000000000192.jpg >data/test.jpeg
