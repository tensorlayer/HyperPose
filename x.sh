#!/bin/sh

set -e

if [ ! -f network-output.npz ]; then
    curl -vLOJ https://github.com/tensorlayer/fast-openpose/files/2374263/network-output.npz.gz
    gzip -d network-output.npz.gz
fi
./idx.py

make
# ./test_openpose_paf.py

# valgrind --leak-check=full ./openpose_paf/cmake-build/$(uname -s)/test_paf 2>val.err.log
./openpose_paf/cmake-build/$(uname -s)/test_paf
