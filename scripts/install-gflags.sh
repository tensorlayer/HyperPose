#!/bin/sh
set -e

cd $(dirname $0)/..
mkdir -p 3rdparty && cd 3rdparty
PREFIX=$(pwd)/local

[ ! -d gflags ] && git clone https://github.com/gflags/gflags.git

cd gflags
mkdir -p cmake-build && cd cmake-build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=${PREFIX}
make && make install
