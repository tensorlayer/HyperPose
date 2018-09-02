#!/usr/bin/env python3
"""Inference with freezed graph."""

import os
import time
import sys

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from estimator2 import TfPoseEstimator as TfPoseEstimator2
from common import read_imgfile, measure, load_graph, get_op, plot_humans

path_to_freezed = 'checkpoints/freezed'


class TfPoseEstimator2Loader(TfPoseEstimator2):

    def __init__(self, path_to_freezed, target_size):
        graph = load_graph(path_to_freezed)

        self.target_size = target_size

        parameter_names = [
            'image',
            'upsample_size',
            'upsample_heatmat',
            'tensor_peaks',
            'upsample_pafmat',
        ]

        for name in parameter_names:
            op = get_op(graph, name)
            print(op)

        (self.tensor_image, self.upsample_size, self.tensor_heatMat_up, self.tensor_peaks,
         self.tensor_pafMat_up) = [get_op(graph, name) for name in parameter_names]

        self._warm_up(graph)

    def _warm_up(self, graph):
        self.persistent_sess = tf.InteractiveSession(graph=graph)
        # self.persistent_sess.run(tf.global_variables_initializer())


def inference(input_files):
    e = measure(lambda: TfPoseEstimator2Loader(path_to_freezed, target_size=(432, 368)),
                'create TfPoseEstimator2Loader')

    for idx, img_name in enumerate(input_files):
        image = read_imgfile(img_name, None, None)
        humans = measure(lambda: e.inference(image, resize_out_ratio=8.0), 'inference')
        print('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                print(h)
            plot_humans(e, image, humans, '%02d' % (idx + 1))


def main():
    input_files = sys.argv[1:]
    if len(input_files) <= 0:
        # TODO: print usage
        exit(1)
    inference(input_files)


if __name__ == '__main__':
    main()
