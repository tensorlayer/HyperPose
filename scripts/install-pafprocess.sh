#!/bin/sh
set -e

cd $(dirname $0)/../openpose_plus/inference

[ ! -d pafprocess ] && svn export https://github.com/ildoonet/tf-pose-estimation/tree/master/tf_pose/pafprocess
cd pafprocess

swig -python -c++ pafprocess.i
python setup.py build_ext --inplace
