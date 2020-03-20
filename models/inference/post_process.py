import numpy as np
import scipy.stats as st
import tensorflow as tf

from . import common
from .common import CocoPart


def _normalize(t):
    return t / t.sum()


def _gauss_kernel(ksize, nsig):
    interval = (2 * nsig + 1.) / ksize
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., ksize + 1)
    y = np.diff(st.norm.cdf(x))
    return _normalize(np.sqrt(np.outer(y, y)))


def _gauss_smooth(origin):
    channels = origin.get_shape().as_list()[3]
    ksize = 25
    gk = _gauss_kernel(ksize, 3.0)
    filters = np.outer(gk, np.ones([channels])).reshape((ksize, ksize, channels, 1))
    return tf.nn.depthwise_conv2d(
        origin, tf.convert_to_tensor(filters, dtype=origin.dtype), [1, 1, 1, 1], padding='SAME')


def get_peak_map(origin, name):
    smoothed = _gauss_smooth(origin)
    max_pooled = tf.nn.pool(smoothed, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    return tf.where(tf.equal(smoothed, max_pooled), smoothed, tf.zeros_like(origin), name)


def upsample(t, upsample_size, name):
    return tf.image.resize_area(t, upsample_size, align_corners=False, name=name)


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
    from .pafprocess import pafprocess  # TODO: don't depend on it
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


class PostProcessor(object):

    def __init__(self, origin_size, feature_size, data_format):
        """Create the PostProcessor.

        Parameters:
            origin_size : (height, width) of the input image
            feature_size : (height', width') of the the feature maps
        """
        self.data_format = data_format

        n_joins = 18 + 1
        n_connections = 17 + 2

        f_height, f_width = feature_size
        self.heatmap_input = tf.placeholder(tf.float32, [1, f_height, f_width, n_joins])
        self.paf_input = tf.placeholder(tf.float32, [1, f_height, f_width, 2 * n_connections])

        self.heapmap_upsample = upsample(self.heatmap_input, origin_size, 'upsample_heatmat')
        self.peaks = get_peak_map(self.heapmap_upsample, 'tensor_peaks')
        self.paf_upsample = upsample(self.paf_input, origin_size, 'upsample_pafmat')

        self.sess = tf.InteractiveSession()

    def __del__(self):
        self.sess.close()

    def __call__(self, heatmap_input, pafmap_input):
        if self.data_format == 'channels_first':
            p = [1, 2, 0]
            heatmap_input = heatmap_input.transpose(p)
            pafmap_input = pafmap_input.transpose(p)

        peaks, heatmap, pafmap = self.sess.run(
            [self.peaks, self.heapmap_upsample, self.paf_upsample],
            feed_dict={
                self.heatmap_input: [heatmap_input],
                self.paf_input: [pafmap_input],
            })

        humans = estimate_paf(peaks[0], heatmap[0], pafmap[0])
        return humans, heatmap[0], pafmap[0]
