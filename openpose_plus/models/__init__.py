import tensorflow as tf

from train_config import config
from ..inference.common import rename_tensor

__all__ = [
    'get_base_model',
    'get_model',
]


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


def get_base_model(name):
    if name == 'vgg':
        from .models_vgg import model
    elif name == 'vggtiny':
        from .models_vggtiny import model
    elif name == 'mobilenet':
        from .models_mobilenet import model
    elif name == 'hao28_experimental':
        from .models_hao28_experimental import model
    else:
        raise RuntimeError('unknown base model %s' % name)
    return model


def get_model(base_model_name):

    def model_func(target_size, data_format):
        base_model = get_base_model(base_model_name)
        n_pos = 19
        image = _input_image(target_size[1], target_size[0], data_format, 'image')
        _, b1_list, b2_list, _ = base_model(image, n_pos, None, None, False, False, data_format=data_format)
        conf_tensor = b1_list[-1].outputs
        pafs_tensor = b2_list[-1].outputs
        with tf.variable_scope('outputs'):
            return image, rename_tensor(conf_tensor, 'conf'), rename_tensor(pafs_tensor, 'paf')

    return model_func


model = get_base_model(config.MODEL.name)
