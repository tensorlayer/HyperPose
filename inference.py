#!/usr/bin/env python3

import os
import time

import numpy as np

import tensorflow as tf
import tensorlayer as tl
from config import config
from inference.pafprocess import pafprocess
from models import model
from utils import draw_results, load_image, get_peak

# TODO: make them flags
input_file = 'data/test.jpeg'
model_file = None
# model_file = 'pose1.npz'

if __name__ == '__main__':
    n_pos = config.MODEL.n_pos
    model_path = config.MODEL.model_path
    h, w = 368, 432  # image size for inferencing, small size can speed up
    if (h % 16 != 0) or (w % 16 != 0):
        raise Exception("image size should be divided by 16")

    # define model
    x = tf.placeholder(tf.float32, [None, h, w, 3], "image")
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

    im = load_image(input_file)

    # inference
    # 1st time need time to compile
    # _, _ = sess.run([conf_tensor, pafs_tensor], feed_dict={x: [im]})
    st = time.time()
    conf, pafs, peak = sess.run([conf_tensor, pafs_tensor, peak_tensor], feed_dict={x: [im]})
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

    humans = estimate_paf(peak[0], conf[0], pafs[0])
    print(humans)

    # draw maps
    draw_results([im], None, conf, None, pafs, None, 'inference')

    # draw connection
