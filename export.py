#!/usr/bin/env python3
"""Export pre-trained openpose model for C++."""

import argparse
import os

import tensorflow as tf
import tensorlayer as tl

from inference.common import measure, rename_tensor
from models import get_base_model_func, get_full_model_func, _input_image

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def mkdir_p(full_path):
    os.makedirs(full_path, exist_ok=True)


def save_graph(sess, checkpoint_dir, name):
    tf.train.write_graph(sess.graph_def, checkpoint_dir, name)


def save_model(sess, checkpoint_dir, global_step=0):
    saver = tf.train.Saver()
    checkpoint_prefix = os.path.join(checkpoint_dir, "saved_checkpoint")
    checkpoint_state_name = 'checkpoint_state'
    saver.save(sess, checkpoint_prefix, global_step=global_step, latest_filename=checkpoint_state_name)


def parse_args():
    parser = argparse.ArgumentParser(description='model exporter')
    parser.add_argument('--base-model', type=str, default='', help='vgg | mobilenet', required=True)
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='checkpoint dir')
    parser.add_argument('--graph-filename', type=str, default='graph.pb.txt', help='graph filename')

    return parser.parse_args()


def get_model_func(base_model_name, full):

    h, w = 368, 432
    target_size = (w, h)
    n_pos = 19

    if full:

        def model_func():

            full_model = get_full_model_func(base_model_name)
            return full_model(n_pos, target_size)

    else:

        def model_func():

            base_model = get_base_model_func(base_model_name)
            data_format = 'channels_last'
            image = _input_image(target_size[1], target_size[0], data_format, 'image')
            _, b1_list, b2_list, _ = base_model(image, n_pos, None, None, False, False, data_format=data_format)
            conf_tensor = b1_list[-1].outputs
            pafs_tensor = b2_list[-1].outputs

            with tf.variable_scope('outputs'):
                return [
                    rename_tensor(conf_tensor, 'conf'),
                    rename_tensor(pafs_tensor, 'paf'),
                ]

    return model_func


def export_model(model_func, checkpoint_dir, path_to_npz, graph_filename):
    mkdir_p(checkpoint_dir)
    model_parameters = model_func()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        measure(lambda: tl.files.load_and_assign_npz_dict(path_to_npz, sess), 'load npz')
        measure(lambda: save_graph(sess, checkpoint_dir, graph_filename), 'save_graph')
        measure(lambda: save_model(sess, checkpoint_dir), 'save_model')

    print('model_parameters:')
    for p in model_parameters:
        print('%s :: %s' % (p.name, p.shape))


def main():
    args = parse_args()
    model_func = get_model_func(args.base_model, False)
    export_model(model_func, args.checkpoint_dir, args.path_to_npz, args.graph_filename)


if __name__ == '__main__':
    main()
