#!/bin/sh
set -e

cd

[ ! -d openpose ] && git clone https://github.com/tensorlayer/openpose.git
cd openpose
git checkout cpp

# TODO: check if it is possible to build without python2
# export PYTHON_BIN_PATH=$(which python3)
make
