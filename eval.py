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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger('openpose-plus')
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

eval_size = -1


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
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    # parser.add_argument('--resize', type=str, default='0x0', help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    # parser.add_argument('--resize-out-ratio', type=float, default=8.0, help='if provided, resize heatmaps before they are post-processed. default=8.0')
    parser.add_argument('--base-model', type=str, default='vgg', help='vgg | mobilenet')
    parser.add_argument('--path-to-npz', type=str, default='', help='path to npz', required=True)
    parser.add_argument('--data-format', type=str, default='channels_last', help='channels_last | channels_first.')
    parser.add_argument('--cocoyear', type=str, default='2017')
    parser.add_argument('--coco-dir', type=str, default='data/mscoco2017/')
    parser.add_argument('--data-idx', type=int, default=-1)
    parser.add_argument('--multi-scale', type=bool, default=False)
    args = parser.parse_args()

    height, width = (368, 432)

    cocoyear_list = ['2014', '2017']
    if args.cocoyear not in cocoyear_list:
        logger.error('cocoyear should be one of %s' % str(cocoyear_list))
        sys.exit(-1)

    # TODO : Scales

    image_dir = args.coco_dir + 'val%s/' % args.cocoyear
    coco_json_file = args.coco_dir + 'annotations/person_keypoints_val%s.json' % args.cocoyear
    cocoGt = COCO(coco_json_file)
    catIds = cocoGt.getCatIds(catNms=['person'])
    keys = cocoGt.getImgIds(catIds=catIds)
    if args.data_idx < 0:
        if eval_size > 0:
            keys = keys[:eval_size]  # only use the first #eval_size elements.
        pass
    else:
        keys = [keys[args.data_idx]]
    logger.info('validation %s set size=%d' % (coco_json_file, len(keys)))
    write_json = '%s.json' % (args.model)

    # logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    # w, h = model_wh(args.resize)
    # if w == 0 or h == 0:
    #     e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    # else:
    #     e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    model_func = get_model(args.base_model)
    estimator = TfPoseEstimator(args.path_to_npz, model_func, target_size=(width, height), data_format=args.data_format)
    plot = True

    result = []
    for i, k in enumerate(tqdm(keys)):
        img_meta = cocoGt.loadImgs(k)[0]
        img_idx = img_meta['id']

        img_name = os.path.join(image_dir, img_meta['file_name'])
        image = read_imgfile(img_name, width, height, data_format=args.data_format)
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
        if args.data_idx >= 0:
            logger.info('score:', k, len(humans), len(anns), avg_score)

            if humans:
                for h in humans:
                    logger.info(h)
            if plot:
                if args.data_format == 'channels_first':
                    image = image.transpose([1, 2, 0])
                plot_humans(image, heatMap, pafMap, humans, '%02d' % (img_idx + 1))

    fp = open(write_json, 'w')
    json.dump(result, fp)
    fp.close()

    cocoDt = cocoGt.loadRes(write_json)
    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.params.imgIds = keys
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print(''.join(["%11.4f |" % x for x in cocoEval.stats]))

    pred = json.load(open(write_json, 'r'))
