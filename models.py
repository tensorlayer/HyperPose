import numpy as np
import scipy.stats as st
import tensorflow as tf
import tensorlayer as tl

from config import config

__all__ = [
    'get_base_model_func',
    'get_full_model_func',
    'full_model',  # the full_model, TODO: deprecated
    'model',  # the base, TODO: deprecated
]


##=========================== for training ===================================
def get_base_model_func(name):
    if name == 'vgg':
        from models_vgg import model
    elif name == 'vggtiny':
        from models_vggtiny import model
    elif name == 'mobilenet':
        from models_mobilenet import model
    else:
        raise RuntimeError('unknown base model %s' % name)
    return model


model = get_base_model_func(config.MODEL.name)


##=========================== for inference ===================================
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


def _input_image(height, width, data_format, name):
    """Create a placeholder for input image."""
    # TODO: maybe make it a Layer in tensorlayer
    if data_format == 'channels_last':
        shape = (None, height, width, 3)
    elif data_format == 'channels_first':
        shape = (None, 3, height, width)
    else:
        raise ValueError('invalid data_format: %s' % data_format)
    return tf.placeholder(tf.float32, shape, name)


def get_full_model_func(base_model_name):

    base_model = get_base_model_func(base_model_name)

    def full_model(n_pos, target_size=(368, 368), data_format='channels_last'):
        """Creates the model including the post processing."""
        image = _input_image(target_size[1], target_size[0], data_format, 'image')
        _, _, _, net = base_model(image, n_pos, False, False, data_format=data_format)

        conf_tensor = tl.layers.get_layers_with_name(net, 'model/stage6/branch1/conf')[0]
        pafs_tensor = tl.layers.get_layers_with_name(net, 'model/stage6/branch2/pafs')[0]

        upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')

        def upsample(t, name):
            return tf.image.resize_area(t, upsample_size, align_corners=False, name=name)

        tensor_heatMat_up = upsample(conf_tensor, 'upsample_heatmat')

        # TODO: consider use named tuple
        return (image, upsample_size, tensor_heatMat_up, _get_peek(tensor_heatMat_up, 'tensor_peaks'),
                upsample(pafs_tensor, 'upsample_pafmat'))

    return full_model


full_model = get_full_model_func(config.MODEL.name)
