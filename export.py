#!/usr/bin/env python3
"""Export pre-trained openpose model for C++."""

import os

import tensorflow as tf
import tensorlayer as tl

from utils import get_peak
from config import config
from models import model

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# TODO: make them flags
input_file = 'data/test.jpeg'
model_file = 'pose1.npz'
checkpoint_dir = 'checkpoints'


def mkdir_p(full_path):
    os.makedirs(full_path, exist_ok=True)


def save_graph(sess):
    mkdir_p(checkpoint_dir)
    tf.train.write_graph(sess.graph_def, 'checkpoints', 'graph.pb.txt')


def save_model(sess, idx=0):
    mkdir_p(checkpoint_dir)
    saver = tf.train.Saver()
    checkpoint_prefix = os.path.join(checkpoint_dir, "saved_checkpoint")
    checkpoint_state_name = 'checkpoint_state'
    saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)


def main():
    n_pos = config.MODEL.n_pos
    model_path = config.MODEL.model_path
    h, w = 368, 432  # image size for inferencing, small size can speed up

    # define model
    x = tf.placeholder(tf.float32, [None, h, w, 3], 'image')
    _, _, _, net = model(x, n_pos, None, None, False, False)

    # get output from network
    conf_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch1/conf')[0]
    pafs_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch2/pafs')[0]
    peak_tensor = get_peak(pafs_tensor)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print('restoring model...')
        tl.files.load_and_assign_npz_dict(os.path.join(model_path, model_file), sess)
        print('restored model...')

        print('saving graph...')
        save_graph(sess)
        print('saved graph')

        print('saving model...')
        save_model(sess)
        print('saved model')


if __name__ == '__main__':
    main()
