#!/bin/sh
set -e

# TF_GIT_URL=https://github.com/tensorflow/tensorflow.git
TF_GIT_URL=https://gitee.com/kuroicrow/tensorflow.git
TF_VERSION=1.10.1
BAZEL_PKG=tensorflow/examples/tf-runner

cd $(dirname $0)
SCRIPT_DIR=$(pwd)
ROOT=$(cd ../.. && pwd)

[ ! -d tensorflow ] && git clone ${TF_GIT_URL}

cd tensorflow
git checkout v${TF_VERSION}
mkdir -p ${BAZEL_PKG}
cd ${BAZEL_PKG}

cp ${ROOT}/src/tf-* .
cp ${ROOT}/src/tracer.h .
cp ${SCRIPT_DIR}/BUILD .

bazel build :all
