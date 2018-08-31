#!/usr/bin/env python3

import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import numpy as np

import tensorflow as tf
import tensorlayer as tl
from config import config
from inference.pafprocess import pafprocess
from models import model
from utils import draw_results, load_image, get_peak

# TODO: make them flags
model_file = None
# model_file = 'pose1.npz'

image_height = 368
image_width = 432


def inference(input_files):
    n_pos = config.MODEL.n_pos
    model_path = config.MODEL.model_path

    # define model
    x = tf.placeholder(tf.float32, [None, image_height, image_width, 3], "image")
    _, _, _, net = model(x, n_pos, None, None, False, False)

    # get output from network
    conf_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch1/conf')[0]
    pafs_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch2/pafs')[0]
    peak_tensor = get_peak(pafs_tensor)

    # restore model parameters
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if model_file:
        tl.files.load_and_assign_npz_dict(os.path.join(model_path, model_file), sess)

    images = [load_image(f) for f in input_files]

    # inference
    # 1st time need time to compile
    # _, _ = sess.run([conf_tensor, pafs_tensor], feed_dict={x: [im]})
    st = time.time()
    conf, pafs, peak = sess.run([conf_tensor, pafs_tensor, peak_tensor], feed_dict={x: images})
    t = time.time() - st
    print("get maps took {}s i.e. {} FPS".format(t, 1. / t))
    # print(conf.shape, pafs.shape, peak.shape)

    # get coordinate results from maps using conf and pafs from network output, and peak
    # using OpenPose's official C++ code for this part
    from inference.estimator import Human

    def estimate_paf(peaks, heat_mat, paf_mat):
        pafprocess.process_paf(peaks, heat_mat, paf_mat)  # C++

        humans = []
        for human_id in range(pafprocess.get_num_humans()):
            human = Human([])
            is_added = False

            for part_idx in range(18):
                c_idx = int(pafprocess.get_part_cid(human_id, part_idx))
                if c_idx < 0:
                    continue

                is_added = True
                human.body_parts[part_idx] = BodyPart('%d-%d' % (human_id, part_idx), part_idx,
                                                      float(pafprocess.get_part_x(c_idx)) / heat_mat.shape[1],
                                                      float(pafprocess.get_part_y(c_idx)) / heat_mat.shape[0],
                                                      pafprocess.get_part_score(c_idx))

            if is_added:
                score = pafprocess.get_score(human_id)
                human.score = score
                humans.append(human)

        return humans

    for a, b, c in zip(peak, conf, pafs):
        humans = estimate_paf(a, b, c)
        print(humans)

    # draw maps
    draw_results(images, None, conf, None, pafs, None, 'inference')

    # TODO: draw connection


if __name__ == '__main__':
    input_files = sys.argv[1:]
    batch_limit = 5
    if len(input_files) > batch_limit:
        print('batch limit is %d' % batch_limit)
        input_files = input_files[:batch_limit]
    inference(input_files)
