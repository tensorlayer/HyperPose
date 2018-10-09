# hao25
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import (BatchNormLayer, ConcatLayer, Conv2d, DepthwiseConv2d, InputLayer, MaxPool2d)

__all__ = [
    'model',
]

W_init = tf.contrib.layers.xavier_initializer()  # tf.truncated_normal_initializer(stddev=0.01)
b_init = None  #tf.constant_initializer(value=0.0)
b_init2 = tf.constant_initializer(value=0.0)
decay = 0.999


def depthwise_conv_block(n, n_filter, filter_size=(3, 3), strides=(1, 1), is_train=False, name="depth_block"):
    with tf.variable_scope(name):
        n = DepthwiseConv2d(n, filter_size, strides, W_init=W_init, b_init=None, name='depthwise')
        n = BatchNormLayer(n, decay=decay, act=tf.nn.relu6, is_train=is_train, name='batchnorm1')
        n = Conv2d(n, n_filter, (1, 1), (1, 1), W_init=W_init, b_init=None, name='conv')
        n = BatchNormLayer(n, decay=decay, act=tf.nn.relu6, is_train=is_train, name='batchnorm2')
    return n


def stage(cnn, b1, b2, n_pos, maskInput1, maskInput2, is_train, name='stageX'):
    """Define the archuecture of stage 2 to 6."""
    with tf.variable_scope(name):
        net = ConcatLayer([cnn, b1, b2], -1, name='concat')
        with tf.variable_scope("branch1"):
            b1 = depthwise_conv_block(net, 128, filter_size=(7, 7), is_train=is_train, name="c1")
            b1 = depthwise_conv_block(b1, 128, filter_size=(7, 7), is_train=is_train, name="c2")
            b1 = depthwise_conv_block(b1, 128, filter_size=(7, 7), is_train=is_train, name="c3")
            b1 = depthwise_conv_block(b1, 128, filter_size=(7, 7), is_train=is_train, name="c4")
            b1 = depthwise_conv_block(b1, 128, filter_size=(7, 7), is_train=is_train, name="c5")
            b1 = depthwise_conv_block(b1, 128, filter_size=(1, 1), is_train=is_train, name="c6")
            b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init2, name='conf')
            if is_train:
                b1.outputs = b1.outputs * maskInput1
        with tf.variable_scope("branch2"):
            b2 = depthwise_conv_block(net, 128, filter_size=(7, 7), is_train=is_train, name="c1")
            b2 = depthwise_conv_block(b2, 128, filter_size=(7, 7), is_train=is_train, name="c2")
            b2 = depthwise_conv_block(b2, 128, filter_size=(7, 7), is_train=is_train, name="c3")
            b2 = depthwise_conv_block(b2, 128, filter_size=(7, 7), is_train=is_train, name="c4")
            b2 = depthwise_conv_block(b2, 128, filter_size=(7, 7), is_train=is_train, name="c5")
            b2 = depthwise_conv_block(b2, 128, filter_size=(1, 1), is_train=is_train, name="c6")
            b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init2, name='pafs')
            if is_train:
                b2.outputs = b2.outputs * maskInput2
    return b1, b2


def model(x, n_pos, mask_miss1, mask_miss2, is_train=False, reuse=None, data_format='channels_last'):  # hao25
    if data_format != 'channels_last':
        # TODO: support NCHW
        print('data_format=%s is ignored' % data_format)

    b1_list = []
    b2_list = []
    with tf.variable_scope('model', reuse):
        x = x - 0.5
        n = InputLayer(x, name='in')
        n = Conv2d(n, 32, (3, 3), (1, 1), None, 'SAME', W_init=W_init, b_init=b_init, name='conv1_1')
        n = BatchNormLayer(n, decay=decay, is_train=is_train, act=tf.nn.relu, name='bn1')
        n = depthwise_conv_block(n, 64, is_train=is_train, name="conv1_depth1")

        n = depthwise_conv_block(n, 128, strides=(2, 2), is_train=is_train, name="conv2_depth1")
        n = depthwise_conv_block(n, 128, is_train=is_train, name="conv2_depth2")
        n1 = n

        n = depthwise_conv_block(n, 256, strides=(2, 2), is_train=is_train, name="conv3_depth1")
        n = depthwise_conv_block(n, 256, is_train=is_train, name="conv3_depth2")
        n2 = n

        n = depthwise_conv_block(n, 512, strides=(2, 2), is_train=is_train, name="conv4_depth1")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="conv4_depth2")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="conv4_depth3")
        n = depthwise_conv_block(n, 512, is_train=is_train, name="conv4_depth4")
        cnn = depthwise_conv_block(n, 512, is_train=is_train, name="conv4_depth5")

        ## low-level features
        # n1 = MaxPool2d(n1, (2, 2), (2, 2), 'same', name='maxpool2d')
        n1 = depthwise_conv_block(n1, 128, strides=(2, 2), is_train=is_train, name="n1_down1")
        n1 = depthwise_conv_block(n1, 128, strides=(2, 2), is_train=is_train, name="n1_down2")
        ## mid-level features
        n2 = depthwise_conv_block(n2, 256, strides=(2, 2), is_train=is_train, name="n2_down1")
        ## combine features
        cnn = ConcatLayer([cnn, n1, n2], -1, name='cancat')

        ## stage1
        with tf.variable_scope("stage1/branch1"):
            b1 = depthwise_conv_block(cnn, 128, filter_size=(7, 7), is_train=is_train, name="c1")
            b1 = depthwise_conv_block(b1, 128, filter_size=(7, 7), is_train=is_train, name="c2")
            b1 = depthwise_conv_block(b1, 128, filter_size=(7, 7), is_train=is_train, name="c3")
            b1 = depthwise_conv_block(b1, 512, filter_size=(1, 1), is_train=is_train, name="c4")
            b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='confs')

            if is_train:
                b1.outputs = b1.outputs * mask_miss1
        with tf.variable_scope("stage1/branch2"):
            b2 = depthwise_conv_block(cnn, 128, filter_size=(7, 7), is_train=is_train, name="c1")
            b2 = depthwise_conv_block(b2, 128, filter_size=(7, 7), is_train=is_train, name="c2")
            b2 = depthwise_conv_block(b2, 128, filter_size=(7, 7), is_train=is_train, name="c3")
            b2 = depthwise_conv_block(b2, 512, filter_size=(1, 1), is_train=is_train, name="c4")
            b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init2, name='pafs')
            if is_train:
                b2.outputs = b2.outputs * mask_miss2
            b1_list.append(b1)
            b2_list.append(b2)

        ## other stages
        # for i in range(2, 7): # [2, 3, 4, 5, 6]
        # for i in [5, 6]:
        for i in [3, 4, 5, 6]:
            b1, b2 = stage(cnn, b1_list[-1], b2_list[-1], n_pos, mask_miss1, mask_miss2, is_train, name='stage%d' % i)
            b1_list.append(b1)
            b2_list.append(b2)
        net = tl.layers.merge_networks([b1_list[-1], b2_list[-1]])
    return cnn, b1_list, b2_list, net
