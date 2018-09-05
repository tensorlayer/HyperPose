# hao18
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import ConcatLayer, Conv2d, InputLayer, MaxPool2d

__all__ = [
    'model',
]

W_init = tf.contrib.layers.xavier_initializer()  # tf.truncated_normal_initializer(stddev=0.01)
b_init = tf.constant_initializer(value=0.0)


def state1(cnn, n_pos, mask_miss1, mask_miss2, is_train):
    """Define the first stage of openpose."""
    with tf.variable_scope("stage1/branch1"):
        b1 = Conv2d(cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
        b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
        b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
        # b1 = Conv2d(b1, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
        b1 = Conv2d(b1, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
        b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='confs')
        if is_train:
            b1.outputs = b1.outputs * mask_miss1
    with tf.variable_scope("stage1/branch2"):
        b2 = Conv2d(cnn, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
        b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
        b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
        # b2 = Conv2d(b2, 512, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
        b2 = Conv2d(b2, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c4')
        b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='pafs')
        if is_train:
            b2.outputs = b2.outputs * mask_miss2
    return b1, b2


def stage2(cnn, b1, b2, n_pos, maskInput1, maskInput2, is_train, scope_name):
    """Define the archuecture of stage 2 and so on."""
    with tf.variable_scope(scope_name):
        net = ConcatLayer([cnn, b1, b2], -1, name='concat')
        with tf.variable_scope("branch1"):
            b1 = Conv2d(net, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
            b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
            b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
            b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c4')
            b1 = Conv2d(b1, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c5')
            b1 = Conv2d(b1, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c6')
            b1 = Conv2d(b1, n_pos, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='conf')
            if is_train:
                b1.outputs = b1.outputs * maskInput1
        with tf.variable_scope("branch2"):
            b2 = Conv2d(net, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c1')
            b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c2')
            b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c3')
            b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c4')
            b2 = Conv2d(b2, 128, (3, 3), (1, 1), tf.nn.relu, 'SAME', W_init=W_init, b_init=b_init, name='c5')
            b2 = Conv2d(b2, 128, (1, 1), (1, 1), tf.nn.relu, 'VALID', W_init=W_init, b_init=b_init, name='c6')
            b2 = Conv2d(b2, 38, (1, 1), (1, 1), None, 'VALID', W_init=W_init, b_init=b_init, name='pafs')
            if is_train:
                b2.outputs = b2.outputs * maskInput2
    return b1, b2


def vgg_network(x):
    x = x - 0.5
    # input layer
    net_in = InputLayer(x, name='input')
    # conv1
    net = Conv2d(
        net_in, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv1_1')
    net = Conv2d(net, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv1_2')
    net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    # conv2
    net = Conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv2_1')
    net = Conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv2_2')
    net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    # conv3
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv3_1')
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv3_2')
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv3_3')
    # net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv3_4') # VGG16 only 3x256
    net = MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool3')
    # conv4
    net = Conv2d(net, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv4_1')
    net = Conv2d(net, 512, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv4_2')
    net = Conv2d(net, 256, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv4_3')
    net = Conv2d(net, 128, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=W_init, b_init=b_init, name='conv4_4')

    return net


def model(x, n_pos, mask_miss1, mask_miss2, is_train=False, reuse=None):
    """Defines the entire pose estimation model."""
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
        # for i in range(2, 7):
        for i in [5, 6]:  # only 3 stage in total
            b1, b2 = stage2(cnn, b1, b2, n_pos, mask_miss1, mask_miss2, is_train, scope_name='stage%d' % i)
            b1_list.append(b1)
            b2_list.append(b2)

        net = tl.layers.merge_networks([b1, b2])
        return cnn, b1_list, b2_list, net
