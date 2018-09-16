#!/bin/sh
set -e

cd $(dirname $0)/../inference

[ ! -d pafprocess ] && svn export https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
cd pafprocess

swig -python -c++ pafprocess.i
python3 setup.py build_ext --inplace
