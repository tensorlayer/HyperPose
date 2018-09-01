#!/usr/bin/env python3

import os
import sys

from common import read_imgfile
from estimator2 import TfPoseEstimator as TfPoseEstimator2


def inference(input_files):
    # desktop = os.path.join(os.getenv('HOME'), 'Desktop')
    # path_to_npz = os.path.join(desktop, 'Log_2108/inf_model255000.npz')
    path_to_npz = ''  # TODO: make it a flag
    e = TfPoseEstimator2(path_to_npz)

    for img_name in input_files:
        image = read_imgfile(img_name, None, None)
        humans = e.inference(image)
        print('got %d humans from %s' % (len(humans), img_name))
        for h in humans:
            print(h)


if __name__ == '__main__':
    input_files = sys.argv[1:]
    batch_limit = 5
    if len(input_files) > batch_limit:
        print('batch limit is %d' % batch_limit)
        input_files = input_files[:batch_limit]
    if len(input_files) <= 0:
        # TODO: print usage
        exit(1)
    inference(input_files)
