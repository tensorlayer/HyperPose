#!/usr/bin/env python3

import os
import sys
import time

from common import read_imgfile
from estimator2 import TfPoseEstimator as TfPoseEstimator2

TRAVIS_CI = os.getenv('TRAVIS') == 'true'


def measure(f, name=None):
    if not name:
        name = f.__name__
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    line = '%s took %fs' % (name, duration)
    print(line)
    with open('profile.log', 'a') as f:
        f.write(line + '\n')
    return result


def inference(input_files):
    desktop = os.path.join(os.getenv('HOME'), 'Desktop')
    path_to_npz = os.path.join(desktop, 'Log_2108/inf_model255000.npz')
    if TRAVIS_CI:
        path_to_npz = ''

    e = measure(lambda: TfPoseEstimator2(path_to_npz, target_size=(432, 368)), 'create TfPoseEstimator2')

    for img_name in input_files:
        image = read_imgfile(img_name, None, None)
        humans = measure(lambda: e.inference(image), 'inference')
        print('got %d humans from %s' % (len(humans), img_name))
        for h in humans:
            print(h)


if __name__ == '__main__':
    input_files = sys.argv[1:]
    if TRAVIS_CI:
        batch_limit = 5
        input_files = input_files[:batch_limit]
    if len(input_files) <= 0:
        # TODO: print usage
        exit(1)
    inference(input_files)
