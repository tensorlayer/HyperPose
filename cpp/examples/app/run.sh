#!/bin/sh
set -e

now_nano() {
    date +%s*1000000000+%N | bc
}

measure() {
    echo "[->] $@ begins"
    local begin=$(now_nano)
    $@
    local end=$(now_nano)
    local duration=$(echo "scale=6; ($end - $begin) / 1000000000" | bc)
    echo "[==] $@ took ${duration}s" | tee -a time.log
}

cd $(dirname $0)
SCRIPT_DIR=$(pwd)

cd ${SCRIPT_DIR}/../../..
ROOT=$(pwd)

TF_ROOT=${ROOT}/cpp/tensorflow

export LD_LIBRARY_PATH=${TF_ROOT}

for f in $(find data/media | grep .jpg); do
    measure ${SCRIPT_DIR}/cmake-build/Darwin/bin/see-pose $f
done
