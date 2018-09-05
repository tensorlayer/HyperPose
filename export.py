#!/usr/bin/env python3
"""Export pre-trained openpose model for C++."""

import argparse
import os

import tensorflow as tf
import tensorlayer as tl

from inference.common import measure
from models import full_model

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# TODO: make them flags
checkpoint_dir = './checkpoints'


def mkdir_p(full_path):
    os.makedirs(full_path, exist_ok=True)


def save_graph(sess):
    mkdir_p(checkpoint_dir)
    tf.train.write_graph(sess.graph_def, checkpoint_dir, 'graph.pb.txt')


def save_model(sess, idx=0):
    mkdir_p(checkpoint_dir)
    saver = tf.train.Saver()
    checkpoint_prefix = os.path.join(checkpoint_dir, "saved_checkpoint")
    checkpoint_state_name = 'checkpoint_state'
    saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)


def parse_args():
    parser = argparse.ArgumentParser(description='model exporter')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    h, w = 368, 432
    target_size = (w, h)
    n_pos = 19

    model_parameters = full_model(n_pos, target_size)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        measure(lambda: tl.files.load_and_assign_npz_dict(args.path_to_npz, sess), 'load npz')
        measure(lambda: save_graph(sess), 'save_graph')
        measure(lambda: save_model(sess), 'save_model')

    print('model_parameters:')
    for p in model_parameters:
        print('%s :: %s' % (p.name, p.shape))


if __name__ == '__main__':
    main()
