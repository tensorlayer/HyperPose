#!/usr/bin/python3
"""Run a tensorflow model using tensorRT."""

import argparse
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from cv2 import imwrite

import tensorrt as trt
import uff
from idx import write_idx
from inference.common import measure, read_imgfile
from models import _input_image, get_model_func
from tensorrt.parsers import uffparser
import pycuda.autoinit as _
import pycuda.driver as cuda

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def _get_model_func(base_model_name):

    h, w = 368, 432
    target_size = (w, h)

    def model_func():
        """Creates the openpose model.

        Returns a pair of lists: (inputs, outputs).
        """
        image, conf, paf = get_model_func(base_model_name)(target_size, 'channels_first')
        return [image], [conf, paf]

    return model_func


def volume(shape):
    v = 1
    for d in shape:
        v *= d
    return v


def infer(engine, x, batch_size):
    n = engine.get_nb_bindings()
    print('%d bindings' % n)

    mems = []  # CPU mem
    d_mems = []  # CUDA mem
    shapes = []
    for i in range(n):
        dims = engine.get_binding_dimensions(i)
        shape = dims.shape()
        print('bind %d :: %s' % (i, shape))
        cnt = volume(shape) * batch_size
        mem = cuda.pagelocked_empty(cnt, dtype=np.float32)
        d_mem = cuda.mem_alloc(cnt * mem.dtype.itemsize)
        shapes.append(shape)
        mems.append(mem)
        d_mems.append(d_mem)

    np.copyto(mems[0], x.flatten())

    stream = cuda.Stream()

    ids = list(range(n))
    inputs_ids = ids[:1]
    outputs_ids = ids[1:]

    for i in inputs_ids:
        cuda.memcpy_htod_async(d_mems[i], mems[i], stream)
    context = engine.create_execution_context()
    context.enqueue(batch_size, [int(p) for p in d_mems], stream.handle, None)
    context.destroy()
    for i in outputs_ids:
        cuda.memcpy_dtoh_async(mems[i], d_mems[i], stream)
    stream.synchronize()
    return [mems[i].reshape(shapes[i]) for i in outputs_ids]


def parse_args():
    parser = argparse.ArgumentParser(description='UFF Runner')
    parser.add_argument(
        '--path-to-npz',
        type=str,
        default=os.path.join(os.getenv('HOME'), 'Downloads/vgg450000_no_cpm.npz'),
        help='path to npz',
        required=False)
    parser.add_argument(
        '--images',
        type=str,
        default='./data/media/COCO_val2014_000000000192.jpg',
        help='comma separated list of image filenames',
        required=False)
    parser.add_argument('--base-model', type=str, default='vgg', help='vgg | mobilenet')
    return parser.parse_args()


def draw_results(image, heats_result, pafs_result, name):
    fig = plt.figure(figsize=(8, 8))
    a = fig.add_subplot(2, 3, 1)
    plt.imshow(image)

    if pafs_result is not None:
        a = fig.add_subplot(2, 3, 3)
        a.set_title('Vectormap result')
        paf_x = np.amax(np.absolute(pafs_result[::2, :, :]), axis=0)
        paf_y = np.amax(np.absolute(pafs_result[1::2, :, :]), axis=0)
        plt.imshow(paf_x, alpha=0.3)
        plt.imshow(paf_y, alpha=0.3)
        plt.colorbar()

    if heats_result is not None:
        a = fig.add_subplot(2, 3, 4)
        a.set_title('Heatmap result')
        tmp = np.amax(heats_result[:-1, :, :], axis=0)
        plt.imshow(tmp, alpha=0.3)
        plt.colorbar()

    plt.savefig(name, dpi=300)


def main():
    args = parse_args()
    height, width, channel = 368, 432, 3
    images = []
    for name in args.images.split(','):
        x = read_imgfile(name, width, height, 'channels_first')  # channels_first is required for tensorRT
        images.append(x)

    model_func = _get_model_func(args.base_model)
    model_inputs, model_outputs = model_func()
    input_names = [p.name[:-2] for p in model_inputs]
    output_names = [p.name[:-2] for p in model_outputs]

    print('input names: %s' % ','.join(input_names))
    print('output names: %s' % ','.join(output_names))  # outputs/conf,outputs/paf

    # with tf.Session() as sess:
    sess = tf.InteractiveSession()
    measure(lambda: tl.files.load_and_assign_npz_dict(args.path_to_npz, sess), 'load npz')
    frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_names)
    tf_model = tf.graph_util.remove_training_nodes(frozen_graph)
    uff_model = measure(lambda: uff.from_tensorflow(tf_model, output_names), 'uff.from_tensorflow')
    print('uff model created')

    parser = uffparser.create_uff_parser()
    inputOrder = 0  # NCHW, https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/_nv_uff_parser_8h_source.html
    parser.register_input(input_names[0], (channel, height, width), inputOrder)
    for name in output_names:
        parser.register_output(name)

    G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
    max_batch_size = 1
    max_workspace_size = 1 << 30
    engine = measure(
        lambda: trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, max_batch_size, max_workspace_size),
        'trt.utils.uff_to_trt_engine')
    print('engine created')

    for idx, x in enumerate(images):
        conf, paf = measure(lambda: infer(engine, x, 1), 'infer')
        draw_results(x.transpose([1, 2, 0]), conf, paf, 'uff-result-%02d.png' % idx)


main()
