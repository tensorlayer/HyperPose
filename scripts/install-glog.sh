#!/bin/sh
set -e

cd $(dirname $0)/..
mkdir -p 3rdparty && cd 3rdparty
PREFIX=$(pwd)/local

[ ! -d glog ] && git clone https://github.com/google/glog.git

cd glog
mkdir -p cmake-build && cd cmake-build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}
make && make install
