#!/usr/bin/env python3

import sys
import os
import logging
import argparse
import json, re

sys.path.append('.')

from openpose_plus.inference.common import measure, plot_humans, read_imgfile
from openpose_plus.inference.estimator import TfPoseEstimator
from openpose_plus.models import get_model
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from train_config import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger('openpose-plus')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def round_int(val):
    return int(round(val))


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


if __name__ == '__main__':

    # TODO : Scales

    image_dir = os.path.join(config.DATA.data_path, 'mscoco%s' % config.DATA.coco_version, 'val%s' % config.DATA.coco_version)
    coco_json_file = os.path.join(config.DATA.data_path, 'mscoco%s' % config.DATA.coco_version, 'annotations/person_keypoints_val%s.json' % config.DATA.coco_version)
    cocoGt = COCO(coco_json_file)
    catIds = cocoGt.getCatIds(catNms=['person'])
    keys = cocoGt.getImgIds(catIds=catIds)
    if config.EVAL.data_idx < 0:
        if config.EVAL.eval_size > 0:
            keys = keys[:config.EVAL.eval_size]  # only use the first #eval_size elements.
        pass
    else:
        keys = [keys[config.EVAL.data_idx]]
    logger.info('validation %s set size=%d' % (coco_json_file, len(keys)))

    height, width = (config.MODEL.win, config.MODEL.hin)
    model_func = get_model(config.MODEL.name)
    estimator = TfPoseEstimator(os.path.join(config.MODEL.model_path, config.EVAL.model), model_func, target_size=(width, height))

    result = []
    for i, k in enumerate(tqdm(keys)):
        img_meta = cocoGt.loadImgs(k)[0]
        img_idx = img_meta['id']

        img_name = os.path.join(image_dir, img_meta['file_name'])
        image = read_imgfile(img_name, width, height)
        if image is None:
            logger.error('image not found, path=%s' % img_name)
            sys.exit(-1)

        # inference the image with the specified network
        humans, heatMap, pafMap = estimator.inference(image)

        scores = 0
        ann_idx = cocoGt.getAnnIds(imgIds=[img_idx], catIds=[1])
        anns = cocoGt.loadAnns(ann_idx)
        for human in humans:
            item = {
                'image_id': img_idx,
                'category_id': 1,
                'keypoints': write_coco_json(human, img_meta['width'], img_meta['height']),
                'score': human.score
            }
            result.append(item)
            scores += item['score']

        avg_score = scores / len(humans) if len(humans) > 0 else 0
        logger.info('image: %s humans: %d anns: %d score: %f' % (img_name, len(humans), len(anns), avg_score))

        if config.EVAL.data_idx >= 0:
            if humans:
                for h in humans:
                    logger.info(h)
        if config.EVAL.plot:
            plot_humans(image, heatMap, pafMap, humans, '%06d' % (img_idx+1))

    write_json = 'eval.json'
    fp = open(write_json, 'w')
    json.dump(result, fp)
    fp.close()

    cocoDt = cocoGt.loadRes(write_json)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = keys
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
