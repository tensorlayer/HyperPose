#!/bin/sh

set -e

cd openpose_paf

./download-testdata.sh

make
./cmake-build/$(uname -s)/test_paf >stdout.log 2>stderr.log
code trace.log
