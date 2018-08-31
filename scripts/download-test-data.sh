#!/bin/sh

set -e

cd $(dirname $0)/..
mkdir -p data

svn export https://github.com/CMU-Perceptual-Computing-Lab/openpose/trunk/examples/media
