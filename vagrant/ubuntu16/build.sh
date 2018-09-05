#!/bin/sh
set -e

cd
git clone https://github.com/tensorlayer/openpose.git
cd openpose
git checkout cpp
make
