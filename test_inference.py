#!/usr/bin/env python3

import argparse
import os
import sys
import time

import tensorflow as tf
import tensorlayer as tl

from inference.common import measure, plot_humans, read_imgfile, _default_profiler, rename_tensor
from inference.estimator2 import TfPoseEstimator as TfPoseEstimator2
from models import get_full_model_func, get_base_model_func, _input_image

tf.logging.set_verbosity(tf.logging.INFO)
tl.logging.set_verbosity(tl.logging.INFO)


def inference(base_model_name, path_to_npz, data_format, input_files, plot):

    def model_func(n_pos, target_size):
        full_model = get_full_model_func(base_model_name)
        return full_model(n_pos, target_size, data_format=data_format)

    height, width = (368, 432)
    e = measure(lambda: TfPoseEstimator2(path_to_npz, model_func, target_size=(width, height)),
                'create TfPoseEstimator2')

    t0 = time.time()
    for idx, img_name in enumerate(input_files):
        image = measure(lambda: read_imgfile(img_name, width, height, data_format=data_format), 'read_imgfile')
        humans = measure(lambda: e.inference(image, resize_out_ratio=8.0), 'e.inference')
        tl.logging.info('got %d humans from %s' % (len(humans), img_name))
        if humans:
            for h in humans:
                tl.logging.debug(h)
            if plot:
                plot_humans(e, image, humans, '%02d' % (idx + 1))
    tot = time.time() - t0
    mean = tot / len(input_files)
    tl.logging.info('inference all took: %f, mean: %f, FPS: %f' % (tot, mean, 1.0 / mean))

def debug_tensor(t, name):
    print('%s :: %s, min: %f, mean: %f, max: %f, std: %f' % (
        name, t.shape, t.min(), t.mean(), t.max(), t.std()))

def inference_base_model(base_model_name, path_to_npz, data_format, input_files, plot):
    """Only run the base model and outputs conf and PAF."""

    def model_func(n_pos, target_size):
        base_model = get_base_model_func(base_model_name)
        data_format = 'channels_last'
        image = _input_image(target_size[1], target_size[0], data_format, 'image')
        _, b1_list, b2_list, _ = base_model(image, n_pos, None, None, False, False, data_format=data_format)
        conf_tensor = b1_list[-1].outputs
        pafs_tensor = b2_list[-1].outputs

        with tf.variable_scope('outputs'):
            return [
                image,
                rename_tensor(conf_tensor, 'conf'),
                rename_tensor(pafs_tensor, 'paf'),
            ]


    height, width = (368, 432)
    target_size = (width, height)
    x, conf, paf = model_func(19, target_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        measure(lambda: tl.files.load_and_assign_npz_dict(path_to_npz, sess), 'load npz')

        t0 = time.time()
        for idx, img_name in enumerate(input_files):
            image = measure(lambda: read_imgfile(img_name, width, height, data_format=data_format), 'read_imgfile')
            c, p = sess.run([conf, paf], {x: [image]})
            debug_tensor(c, 'conf tensor')
            debug_tensor(p, 'PAF')
            from idx import write_idx
            write_idx('conf.idx', c[0])
            write_idx('paf.idx', p[0])

def parse_args():
    parser = argparse.ArgumentParser(description='inference')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--images', type=str, default='', help='comma separate list of image filenames', required=True)
    parser.add_argument('--base-model', type=str, default='vgg', help='vgg | mobilenet')
    parser.add_argument('--data-format', type=str, default='channels_last', help='channels_last | channels_first.')
    parser.add_argument('--plot', type=bool, default=False, help='draw the results')
    parser.add_argument('--repeat', type=int, default=1, help='repeat the images for n times for profiling.')
    parser.add_argument('--limit', type=int, default=100, help='max number of images.')

    return parser.parse_args()


def main():
    args = parse_args()
    image_files = ([f for f in args.images.split(',') if f] * args.repeat)[:args.limit]
    # inference(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot)
    inference_base_model(args.base_model, args.path_to_npz, args.data_format, image_files, args.plot)


if __name__ == '__main__':
    measure(main)
    _default_profiler.report()
