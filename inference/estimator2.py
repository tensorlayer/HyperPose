import logging
import math
import time

import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

from inference import common
from inference.common import CocoPart

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class Human:
    """
    body_parts: list of BodyPart
    """
    __slots__ = ('body_parts', 'pairs', 'uidx_list', 'score')

    def __init__(self, pairs):
        self.pairs = []
        self.uidx_list = set()
        self.body_parts = {}
        for pair in pairs:
            self.add_pair(pair)
        self.score = 0.0

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()


class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """
    __slots__ = ('uidx', 'part_idx', 'x', 'y', 'score')

    def __init__(self, uidx, part_idx, x, y, score):
        self.uidx = uidx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.score = score

    def get_part_name(self):
        return CocoPart(self.part_idx)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()


def estimate_paf(peaks, heat_mat, paf_mat):
    from inference.pafprocess import pafprocess  # TODO: don't depend on it
    pafprocess.process_paf(peaks, heat_mat, paf_mat)

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
            human.score = pafprocess.get_score(human_id)
            humans.append(human)

    return humans


class TfPoseEstimator:

    def __init__(self, graph_path, model_func, target_size=(368, 368), tf_config=None):
        n_pos = 19
        self.target_size = target_size
        (self.tensor_image, self.upsample_size, self.tensor_heatMat_up, self.tensor_peaks,
         self.tensor_pafMat_up) = model_func(n_pos, target_size)
        self._warm_up(graph_path)
        logger.info('tensor_heatMat_up :: %s, tensor_pafMat_up :: %s' % (self.tensor_heatMat_up.shape,
                                                                         self.tensor_pafMat_up.shape))

    def _warm_up(self, graph_path):
        self.persistent_sess = tf.InteractiveSession()
        self.persistent_sess.run(tf.global_variables_initializer())
        tl.files.load_and_assign_npz_dict(graph_path, self.persistent_sess)

    def __del__(self):
        # self.persistent_sess.close()
        pass

    @staticmethod
    def _quantize_img(npimg):
        npimg_q = npimg + 1.0
        npimg_q /= (2.0 / 2**8)
        # npimg_q += 0.5
        npimg_q = npimg_q.astype(np.uint8)
        return npimg_q

    def inference(self, npimg, resize_to_default=True, resize_out_ratio=1.0):
        upsample_size = [
            int(self.target_size[1] / 8 * resize_out_ratio),
            int(self.target_size[0] / 8 * resize_out_ratio)
        ]

        if self.tensor_image.dtype == tf.quint8:
            # quantize input image
            npimg = TfPoseEstimator._quantize_img(npimg)

        logger.debug('inference+ upsample_size %s' % (upsample_size,))
        logger.debug('inference+ original shape=%s' % (npimg.shape,))

        peaks, heatMat_up, pafMat_up = self.persistent_sess.run(
            [self.tensor_peaks, self.tensor_heatMat_up, self.tensor_pafMat_up],
            feed_dict={
                self.tensor_image: [npimg],
                self.upsample_size: upsample_size
            })

        t = time.time()
        humans = estimate_paf(peaks[0], heatMat_up[0], pafMat_up[0])
        logger.info('estimate time=%.5f' % (time.time() - t))
        return humans, heatMat_up[0], pafMat_up[0]
