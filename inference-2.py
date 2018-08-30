#!/usr/bin/env python3
"""Inference with freezed graph."""

import os
import time

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from inference.pafprocess import pafprocess
from utils import draw_results, load_image
from debug import tensor_summary

model_file = 'checkpoints/freezed'
input_file = 'data/test.jpeg'


def load_graph(model_file):
    """Load a freezed graph from file."""
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph


def get_op(graph, name):
    return graph.get_operation_by_name('import/%s' % name).outputs[0]


def run(graph):
    input_name = 'image'
    output_names = [
        'model/cpm/stage6/branch1/conf/BiasAdd',
        'model/cpm/stage6/branch2/pafs/BiasAdd',
        'Select',
    ]

    input_operation = get_op(graph, input_name)
    output_operations = [get_op(graph, name) for name in output_names]
    im = load_image(input_file)

    with tf.Session(graph=graph) as sess:
        conf, pafs, peak = sess.run(
            output_operations,
            {
                input_operation: [im],
            },
        )

    tensor_summary('conf', conf)
    tensor_summary('pafs', pafs)
    tensor_summary('peak', peak)

    # draw maps
    draw_results([im], None, conf, None, pafs, None, 'inference')


def main():
    g = load_graph(model_file)

    st = time.time()
    run(g)
    t = time.time() - st
    print("get maps took {}s i.e. {} FPS".format(t, 1. / t))


main()
