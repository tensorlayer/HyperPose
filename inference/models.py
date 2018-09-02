import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import ConcatLayer, Conv2d, InputLayer, MaxPool2d

from tensblur.smoother import Smoother

__all__ = [
    'full_model',
    'model',
]

W_init = tf.truncated_normal_initializer(stddev=0.01)
b_init = tf.constant_initializer(value=0.0)


def state1(cnn, n_pos, mask_miss1, mask_miss2, is_train):
    """Define the first stage of openpose."""
    with tf.variable_scope("stage1/branch1"):
        b1 = Conv2d(cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
        b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
        b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
        b1 = Conv2d(b1, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
        b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='confs')
        if is_train:
            b1.outputs = b1.outputs * mask_miss1
    with tf.variable_scope("stage1/branch2"):
        b2 = Conv2d(cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
        b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
        b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
        b2 = Conv2d(b2, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
        b2 = Conv2d(b2, n_pos * 2, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='pafs')
        if is_train:
            b2.outputs = b2.outputs * mask_miss2
    return b1, b2


def stage2(cnn, b1, b2, n_pos, maskInput1, maskInput2, is_train, scope_name):
    """Define the archuecture of stage 2 and so on."""
    with tf.variable_scope(scope_name):
        net = ConcatLayer([cnn, b1, b2], -1, name='concat')
        with tf.variable_scope("branch1"):
            b1 = Conv2d(net, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c4')
            b1 = Conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c5')
            b1 = Conv2d(b1, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c6')
            b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='conf')
            if is_train:
                b1.outputs = b1.outputs * maskInput1
        with tf.variable_scope("branch2"):
            b2 = Conv2d(net, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c4')
            b2 = Conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c5')
            b2 = Conv2d(b2, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c6')
            b2 = Conv2d(b2, n_pos * 2, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='pafs')
            if is_train:
                b2.outputs = b2.outputs * maskInput2
    return b1, b2


def vgg_network(x):
    """VGG19 network for default model."""

    # input x: 0~1
    # bgr: -0.5~0.5 for InputLayer
    red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
    bgr = tf.concat(axis=3, values=[blue, green, red])
    bgr = bgr - 0.5

    # input layer
    net_in = InputLayer(bgr, name='input')
    # conv1
    net = Conv2d(net_in, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
    net = Conv2d(net, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
    net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    # conv2
    net = Conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
    net = Conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
    net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    # conv3
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
    net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool3')
    # conv4
    net = Conv2d(net, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
    net = Conv2d(net, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv4_3')
    net = Conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv4_4')

    return net


def model(x, n_pos, mask_miss1, mask_miss2, is_train=False, reuse=None):
    """Defines the entire pose estimation model."""
    with tf.variable_scope('model', reuse):
        # Feature extraction part
        # 1. by default, following the paper, we use VGG19 as the default model
        cnn = vgg_network(x)
        # cnn = tl.models.VGG19(x, end_with='conv4_2', reuse=reuse)
        # 2. you can customize this part to speed up the inferencing
        # cnn = tl.models.MobileNetV1(x, end_with='depth5', is_train=is_train, reuse=reuse)  # i.e. vgg16 conv4_2 ~ 4_4

        b1_list = []
        b2_list = []
        with tf.variable_scope('cpm', reuse):
            # stage 1
            b1, b2 = state1(cnn, n_pos, mask_miss1, mask_miss2, is_train)
            b1_list.append(b1)
            b2_list.append(b2)
            # stage 2 ~ 6
            for i in range(2, 7):
                b1, b2 = stage2(cnn, b1, b2, n_pos, mask_miss1, mask_miss2, is_train, scope_name='stage%d' % i)
                b1_list.append(b1)
                b2_list.append(b2)

        net = tl.layers.merge_networks([b1, b2])
        return cnn, b1_list, b2_list, net


def _get_peek(tensor, name):
    smoother = Smoother({'data': tensor}, 25, 3.0)
    gaussian_heatMat = smoother.get_output()
    max_pooled_in_tensor = tf.nn.pool(gaussian_heatMat, window_shape=(3, 3), pooling_type='MAX', padding='SAME')
    return tf.where(
        tf.equal(gaussian_heatMat, max_pooled_in_tensor), gaussian_heatMat, tf.zeros_like(gaussian_heatMat), name)


def full_model(n_pos, target_size=(368, 368)):
    """Creates the model including the post processing."""
    image = tf.placeholder(tf.float32, [None, target_size[1], target_size[0], 3], "image")
    _, _, _, net = model(image, n_pos, False, False)

    conf_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch1/conf')[0]
    pafs_tensor = tl.layers.get_layers_with_name(net, 'model/cpm/stage6/branch2/pafs')[0]

    upsample_size = tf.placeholder(dtype=tf.int32, shape=(2,), name='upsample_size')

    def upsample(t, name):
        return tf.image.resize_area(t, upsample_size, align_corners=False, name=name)

    tensor_heatMat_up = upsample(conf_tensor, 'upsample_heatmat')

    # TODO: consider use named tuple
    return (image, upsample_size, tensor_heatMat_up, _get_peek(tensor_heatMat_up, 'tensor_peaks'),
            upsample(pafs_tensor, 'upsample_pafmat'))
