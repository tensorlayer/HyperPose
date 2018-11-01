#!/usr/bin/env python3
"""Export pre-trained openpose model for C++/TensorRT."""

import argparse
import os
import sys

import tensorflow as tf
import tensorlayer as tl

sys.path.append('.')

from openpose_plus.inference.common import measure, rename_tensor
from openpose_plus.models import get_model

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


def save_uff(sess, names, filename):
    import uff
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, names)
    tf_model = tf.graph_util.remove_training_nodes(frozen_graph)
    uff.from_tensorflow(tf_model, names, output_filename=filename)


def export_model(model_func, checkpoint_dir, path_to_npz, graph_filename, uff_filename):
    mkdir_p(checkpoint_dir)
    model_parameters = model_func()
    names = [p.name[:-2] for p in model_parameters]
    print('name: %s' % ','.join(names))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        measure(lambda: tl.files.load_and_assign_npz_dict(path_to_npz, sess), 'load npz')

        if graph_filename:
            measure(lambda: save_graph(sess, checkpoint_dir, graph_filename), 'save_graph')
            measure(lambda: save_model(sess, checkpoint_dir), 'save_model')

        if uff_filename:
            measure(lambda: save_uff(sess, names, uff_filename), 'save_uff')

    print('exported model_parameters:')
    for p in model_parameters:
        print('%s :: %s' % (p.name, p.shape))


def parse_args():
    parser = argparse.ArgumentParser(description='model exporter')
    parser.add_argument('--base-model', type=str, default='', help='vgg | vggtiny | mobilenet', required=True)
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='checkpoint dir')
    parser.add_argument('--graph-filename', type=str, default='', help='graph filename')
    parser.add_argument('--uff-filename', type=str, default='', help='uff filename')
    parser.add_argument('--data-format', type=str, default='channels_last', help='channels_last | channels_first.')
    parser.add_argument('--height', type=int, default='368', help='input height.')
    parser.add_argument('--width', type=int, default='432', help='input width.')

    return parser.parse_args()


def main():
    args = parse_args()

    def model_func():
        target_size = (args.width, args.height)
        return get_model(args.base_model)(target_size, args.data_format)

    export_model(model_func, args.checkpoint_dir, args.path_to_npz, args.graph_filename, args.uff_filename)


if __name__ == '__main__':
    main()
