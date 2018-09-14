#!/bin/sh
set -e

cd $(dirname $0)

[ ! -d pafprocess ] && svn export https://github.com/ildoonet/tf-pose-estimation/trunk/tf_pose/pafprocess
