#!/usr/bin/env python3

import argparse
import os
import sys
import time

from inference.common import measure, plot_humans, read_imgfile
from inference.estimator2 import TfPoseEstimator as TfPoseEstimator2
from models import full_model


def inference(path_to_npz, input_files):
    e = measure(lambda: TfPoseEstimator2(path_to_npz, full_model, target_size=(432, 368)), 'create TfPoseEstimator2')

    for idx, img_name in enumerate(input_files):
        image = read_imgfile(img_name, None, None)
        humans = measure(lambda: e.inference(image, resize_out_ratio=8.0), 'inference')
        print('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                print(h)
            plot_humans(e, image, humans, '%02d' % (idx + 1))


def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--images', type=str, default='', help='comma separate list of image filenames', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    image_files = [f for f in args.images.split(',') if f]
    inference(args.path_to_npz, image_files)
