#!/bin/sh
set -e

make
otool -L cmake-build/Darwin/tf-runner_main
./cmake-build/$(uname -s)/tf-runner_main
