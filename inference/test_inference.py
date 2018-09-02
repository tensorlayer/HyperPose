#!/usr/bin/env python3

import os
import sys
import time

from common import read_imgfile, measure, plot_humans

from estimator2 import TfPoseEstimator as TfPoseEstimator2

TRAVIS_CI = os.getenv('TRAVIS') == 'true'


def inference(input_files):
    desktop = os.path.join(os.getenv('HOME'), 'Desktop')
    path_to_npz = os.path.join(desktop, 'Log_2108/inf_model255000.npz')
    if TRAVIS_CI:
        path_to_npz = ''

    e = measure(lambda: TfPoseEstimator2(path_to_npz, target_size=(432, 368)), 'create TfPoseEstimator2')

    for idx, img_name in enumerate(input_files):
        image = read_imgfile(img_name, None, None)
        humans = measure(lambda: e.inference(image, resize_out_ratio=8.0), 'inference')
        print('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                print(h)
            plot_humans(e, image, humans, '%02d' % (idx + 1))


if __name__ == '__main__':
    input_files = sys.argv[1:]
    if TRAVIS_CI:
        batch_limit = 5
        input_files = input_files[:batch_limit]
    if len(input_files) <= 0:
        # TODO: print usage
        exit(1)
    inference(input_files)
