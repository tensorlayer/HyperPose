import logging
import time

import tensorflow as tf
import tensorlayer as tl

from .post_process import PostProcessor

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class TfPoseEstimator:

    def __init__(self, graph_path, model_func, target_size=(368, 368), data_format='channels_last'):
        width, height = target_size
        f_height, f_width = (height / 8, width / 8)
        self.post_processor = PostProcessor((height, width), (f_height, f_width), data_format)

        self.tensor_image, self.tensor_heatmap, self.tensor_paf = model_func(target_size, data_format)
        self._warm_up(graph_path)

    def _warm_up(self, graph_path):
        self.persistent_sess = tf.InteractiveSession()
        self.persistent_sess.run(tf.global_variables_initializer())
        tl.files.load_and_assign_npz_dict(graph_path, self.persistent_sess)

    def __del__(self):
        self.persistent_sess.close()

    def inference(self, npimg):
        heatmap, pafmap = self.persistent_sess.run(
            [self.tensor_heatmap, self.tensor_paf], feed_dict={
                self.tensor_image: [npimg],
            })

        t = time.time()
        humans, heatmap_up, pafmap_up = self.post_processor(heatmap[0], pafmap[0])
        logger.info('estimate time=%.5f' % (time.time() - t))
        return humans, heatmap_up, pafmap_up
