#!/bin/sh
set -e

paf_install_path="$(dirname $0)/../openpose_plus/inference"
[ ! -d $paf_install_path ] && mkdir $paf_install_path
cd $paf_install_path

[ ! -d pafprocess ] && svn export https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
cd pafprocess

swig -python -c++ pafprocess.i
python setup.py build_ext --inplace
