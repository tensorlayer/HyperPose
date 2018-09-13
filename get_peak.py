import numpy as np
import scipy.stats as st
import tensorflow as tf
import tensorlayer as tl


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


def _get_peek(origin, name):
    smoothed = _gauss_smooth(origin)
    max_pooled = tf.nn.pool(smoothed, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    return tf.where(tf.equal(smoothed, max_pooled), smoothed, tf.zeros_like(origin), name)
