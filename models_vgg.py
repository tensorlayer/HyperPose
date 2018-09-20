import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import ConcatLayer, Conv2d, InputLayer, MaxPool2d

__all__ = [
    'model',
]

W_init = tf.contrib.layers.xavier_initializer()  # tf.truncated_normal_initializer(stddev=0.01)
b_init = tf.constant_initializer(value=0.0)


def model(x, n_pos, mask_miss1, mask_miss2, is_train=False, reuse=None, data_format='channels_last'):
    """Defines the entire pose estimation model."""

    def _conv2d(x, c, filter_size, strides, act, padding, name):
        return Conv2d(
            x, c, filter_size, strides, act, padding, W_init=W_init, b_init=b_init, name=name, data_format=data_format)

    def _maxpool2d(x, name):
        return MaxPool2d(x, (2, 2), (2, 2), padding='SAME', name=name, data_format=data_format)

    def state1(cnn, n_pos, mask_miss1, mask_miss2, is_train):
        """Define the first stage of openpose."""

        with tf.variable_scope("stage1/branch1"):
            b1 = _conv2d(cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c1')
            b1 = _conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c2')
            b1 = _conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c3')
            b1 = _conv2d(b1, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', 'c4')
            b1 = _conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', 'confs')
            if is_train:
                b1.outputs = b1.outputs * mask_miss1
        with tf.variable_scope("stage1/branch2"):
            b2 = _conv2d(cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c1')
            b2 = _conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c2')
            b2 = _conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', 'c3')
            b2 = _conv2d(b2, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', 'c4')
            b2 = _conv2d(b2, n_pos * 2, (1, 1), (1, 1), None, 'VALID', 'pafs')
            if is_train:
                b2.outputs = b2.outputs * mask_miss2
        return b1, b2

    def stage2(cnn, b1, b2, n_pos, maskInput1, maskInput2, is_train, scope_name):
        """Define the archuecture of stage 2 and so on."""

        with tf.variable_scope(scope_name):
            if data_format == 'channels_last':
                concat_dim = -1
            elif data_format == 'channels_first':
                concat_dim = 1
            else:
                raise ValueError('invalid data_format: %s' % data_format)
            net = ConcatLayer([cnn, b1, b2], concat_dim, name='concat')

            with tf.variable_scope("branch1"):
                b1 = _conv2d(net, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c1')
                b1 = _conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c2')
                b1 = _conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c3')
                b1 = _conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c4')
                b1 = _conv2d(b1, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c5')
                b1 = _conv2d(b1, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', 'c6')
                b1 = _conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', 'conf')
                if is_train:
                    b1.outputs = b1.outputs * maskInput1
            with tf.variable_scope("branch2"):
                b2 = _conv2d(net, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c1')
                b2 = _conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c2')
                b2 = _conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c3')
                b2 = _conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c4')
                b2 = _conv2d(b2, 128, (7, 7), (1, 1), tf.nn.relu, 'SAME', 'c5')
                b2 = _conv2d(b2, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', 'c6')
                b2 = _conv2d(b2, n_pos * 2, (1, 1), (1, 1), None, 'VALID', 'pafs')
                if is_train:
                    b2.outputs = b2.outputs * maskInput2
        return b1, b2

    def vgg_network(x):
        """VGG19 network for default model."""

        # input x: 0~1
        # bgr: -0.5~0.5 for InputLayer
        # red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=x)
        # bgr = tf.concat(axis=3, values=[blue, green, red])
        # bgr = bgr - 0.5

        # input layer
        net_in = InputLayer(x, name='input')

        # conv1
        net = _conv2d(net_in, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        net = _conv2d(net, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        net = _maxpool2d(net, 'pool1')
        # conv2
        net = _conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        net = _conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        net = _maxpool2d(net, 'pool2')
        # conv3
        net = _conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        net = _conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        net = _conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        net = _conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        net = _maxpool2d(net, 'pool3')
        # conv4
        net = _conv2d(net, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        net = _conv2d(net, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        net = _conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        net = _conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')

        return net

    with tf.variable_scope('model', reuse):
        ## Feature extraction part
        cnn = vgg_network(x)
        b1_list = []
        b2_list = []
        ## stage 1
        b1, b2 = state1(cnn, n_pos, mask_miss1, mask_miss2, is_train)
        b1_list.append(b1)
        b2_list.append(b2)
        ## stage 2 ~ 6
        for i in range(2, 7):
            b1, b2 = stage2(cnn, b1, b2, n_pos, mask_miss1, mask_miss2, is_train, 'stage%d' % i)
            b1_list.append(b1)
            b2_list.append(b2)

        net = tl.layers.merge_networks([b1, b2])
        return cnn, b1_list, b2_list, net
