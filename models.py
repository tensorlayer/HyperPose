import tensorflow as tf
import tensorlayer as tl
from config import config
from inference.tensblur.smoother import Smoother

__all__ = [
    'get_bsae_model',
    'full_model',  # the full_model, TODO: deprecated
    'create_full_model',
]


def get_bsae_model(name):
    if name == 'vgg':
        from models_vgg import model
    elif name == 'mobilenet':
        from models_mobilenet import model
    return model


def _get_peek(tensor, name):
    smoother = Smoother({'data': tensor}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()
    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    return tf.where(
        tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat, tf.zeros_like(gaussian_heatMat), name)


def create_full_model(base_model):

    def full_model(n_pos, target_size=(368, 368)):
        """Creates the model including the post processing."""
        image = tf.placeholder(tf.float32, [None, target_size[1], target_size[0], 3], "image")
        _, _, _, net = base_model(image, n_pos, False, False)

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


full_model = create_full_model(get_bsae_model(config.MODEL.name))
