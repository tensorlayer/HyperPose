#!/usr/bin/env python3

import argparse
import json
import logging
import os
import re
import sys

import numpy as np
from tqdm import tqdm

from inference.common import plot_humans, read_imgfile
from inference.estimator2 import TfPoseEstimator as TfPoseEstimator2
from models import full_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_size = 100


def round_int(val):
    return int(round(val))


def model_wh(resolution_str):
    width, height = map(int, resolution_str.split('x'))
    if width % 16 != 0 or height % 16 != 0:
        raise Exception('Width and height should be multiples of 16. w=%d, h=%d' % (width, height))
    return int(width), int(height)


def write_coco_json(human, image_w, image_h):
    keypoints = []
    coco_ids = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
    for coco_id in coco_ids:
        if coco_id not in human.body_parts.keys():
            keypoints.extend([0, 0, 0])
            continue
        body_part = human.body_parts[coco_id]
        keypoints.extend([round_int(body_part.x * image_w), round_int(body_part.y * image_h), 2])
    return keypoints


def parse_args():
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument(
        '--resize',
        type=str,
        default='432x368',
        help=
        'if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 '
    )
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')
    parser.add_argument('--cocoyear', type=str, default='2017')
    parser.add_argument('--coco-dir', type=str, default='')
    parser.add_argument('--data-idx', type=int, default=-5)
    parser.add_argument('--multi-scale', type=bool, default=True)
    parser.add_argument('--net_type', type=str, default='full_normal')
    parser.add_argument('--base_dir', type=str, default='./data/media')
    parser.add_argument('--path-to-npz', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    w, h = model_wh(args.resize)

    if args.net_type == 'full_normal':
        e = TfPoseEstimator2(args.path_to_npz, full_model, target_size=(w, h))

    imglist = os.listdir(args.base_dir)
    for idx, image_name in enumerate(imglist):
        img_name = os.path.join(args.base_dir, image_name)
        image = read_imgfile(img_name, w, h)
        humans, heatMap, pafMap = e.inference(image)
        plot_humans(image, heatMap, pafMap, humans, '%02d' % (idx + 1))
