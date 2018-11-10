# hao28 --> MobileNetV2
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import ConcatLayer, Conv2d, InputLayer, MaxPool2d, BatchNormLayer, PReluLayer, DepthwiseConv2d, ElementwiseLayer

__all__ = ['model']

# W_init = tf.contrib.layers.xavier_initializer()
W_init = tf.truncated_normal_initializer(stddev=0.01)
# b_init = None # tf.constant_initializer(value=0.0)
b_init2 = tf.constant_initializer(value=0.0)
decay = 0.999

tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)


def mobilenetv2_bottleneck(nin, n_filter, filter_size=(3, 3), strides=(1, 1), t=4, is_train=False, is_res=True, data_format='channel_last', name='bottleneck'):
    """MobileNet V2 Bottleneck with P-ReLU.

    Parameters
    -----------
    t : int
        Expansion factor for mobilenet v2.
    is_res : boolean
        If True, add input to the output.
    """
    input_shape = nin.outputs.get_shape().as_list()
    if data_format=='channel_last':
        n_filter_current = input_shape[-1]
    elif data_format=='channel_first':
        n_filter_current = input_shape[0]
    else:
        raise Exception("unknown data_format")

    with tf.variable_scope(name):
        n = Conv2d(nin, n_filter_current * t, (1, 1), (1, 1), act=None, padding='SAME', W_init=W_init, b_init=None, name='conv1x1')
        n = BatchNormLayer(n, decay=0.999, act=None, is_train=is_train, name='conv1x1/bn')
        n = PReluLayer(n, name='conv1x1/prelu')

        n = DepthwiseConv2d(n, filter_size, strides, W_init=W_init, b_init=None, name='depthwiseconv')
        n = BatchNormLayer(n, decay=0.999, act=None, is_train=is_train, name='depthwiseconv/bn')
        n = PReluLayer(n, name='depthwiseconv/prelu')

        n = Conv2d(n, n_filter, (1, 1), (1, 1), act=None, padding='SAME', W_init=W_init, b_init=None, name='conv1x1out')
        n = BatchNormLayer(n, decay=0.999, act=None, is_train=is_train, name='conv1x1out/bn')

        if is_res:
            n = ElementwiseLayer([nin, n], tf.add, name='resadd')
    return n

def model(x, n_pos, mask_miss1, mask_miss2, is_train=False, reuse=None, data_format='channels_last'):
    """Defines the entire pose estimation model."""
    def _conv2d(x, c, filter_size, strides, act, padding, name):
        return Conv2d(
            x, c, filter_size, strides, act, padding, W_init=W_init, b_init=None, name=name, data_format=data_format)

    def _conv2d_with_bias_init(x, c, filter_size, strides, act, padding, name):
        return Conv2d(
            x, c, filter_size, strides, act, padding, W_init=W_init, b_init=b_init2, name=name, data_format=data_format)

    def _maxpool2d(x, name):
        return MaxPool2d(x, (2, 2), (2, 2), padding='SAME', name=name, data_format=data_format)

    def bn(x, name):
        return BatchNormLayer(
            x,
            is_train=is_train,
            act=tf.nn.relu,
            decay=decay,
            name=name,
            # https://github.com/tensorlayer/tensorlayer/commit/4e6f768bd2d0c0f27c2385ce7f541b848deb7953
            data_format=data_format)

    def concat(inputs, name):
        if data_format == 'channels_last':
            concat_dim = -1
        elif data_format == 'channels_first':
            concat_dim = 1
        else:
            raise ValueError('invalid data_format: %s' % data_format)
        return ConcatLayer(inputs, concat_dim, name=name)

    def stage(cnn, b1, b2, n_pos, maskInput1, maskInput2, is_train, name='stageX'):
        """Define the archuecture of stage 2 to 6."""
        with tf.variable_scope(name):
            net = concat([cnn, b1, b2], 'concat')
            with tf.variable_scope("branch1"):
                b1 = _conv2d(net, 128, (3, 3), (1, 1), None, 'SAME', 'c1')
                b1 = bn(b1, 'bn1')
                b1 = _conv2d(b1, 128, (3, 3), (1, 1), None, 'SAME', 'c2')
                b1 = bn(b1, 'bn2')
                b1 = _conv2d(b1, 128, (3, 3), (1, 1), None, 'SAME', 'c3')
                b1 = bn(b1, 'bn3')
                b1 = _conv2d(b1, 128, (3, 3), (1, 1), None, 'SAME', 'c4')
                b1 = bn(b1, 'bn4')
                b1 = _conv2d(b1, 128, (3, 3), (1, 1), None, 'SAME', 'c5')
                b1 = bn(b1, 'bn5')
                b1 = _conv2d(b1, 128, (1, 1), (1, 1), None, 'VALID', 'c6')
                b1 = bn(b1, 'bn6')
                b1 = _conv2d_with_bias_init(b1, n_pos, (1, 1), (1, 1), None, 'VALID', 'conf')
                if is_train:
                    b1.outputs = b1.outputs * maskInput1
            with tf.variable_scope("branch2"):
                b2 = _conv2d(net, 128, (3, 3), (1, 1), None, 'SAME', 'c1')
                b2 = bn(b2, 'bn1')
                b2 = _conv2d(b2, 128, (3, 3), (1, 1), None, 'SAME', 'c2')
                b2 = bn(b2, 'bn2')
                b2 = _conv2d(b2, 128, (3, 3), (1, 1), None, 'SAME', 'c3')
                b2 = bn(b2, 'bn3')
                b2 = _conv2d(b2, 128, (3, 3), (1, 1), None, 'SAME', 'c4')
                b2 = bn(b2, 'bn4')
                b2 = _conv2d(b2, 128, (3, 3), (1, 1), None, 'SAME', 'c5')
                b2 = bn(b2, 'bn5')
                b2 = _conv2d(b2, 128, (1, 1), (1, 1), None, 'VALID', 'c6')
                b2 = bn(b2, 'bn6')
                b2 = _conv2d_with_bias_init(b2, 38, (1, 1), (1, 1), None, 'VALID', name='pafs')
                if is_train:
                    b2.outputs = b2.outputs * maskInput2
        return b1, b2

    def cnn_net(x, is_train):
        """Simplified VGG19 network for default model."""

        # input x: 0~1
        x = x - 0.5

        # input layer
        n = InputLayer(x, name='input')
        # conv1
        # net = _conv2d(net_in, 32, (3, 3), (1, 1), None, 'SAME', 'conv1_1')
        # net = bn(net, 'bn1_1')
        # net = _conv2d(net, 64, (3, 3), (1, 1), None, 'SAME', 'conv1_2')
        # net = bn(net, 'bn1_2')
        n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=W_init, b_init=None, name='conv1')
        n = BatchNormLayer(n, decay=0.999, act=None, is_train=is_train, name='conv1/bn')
        n = PReluLayer(n, name='conv1/prelu')
            # n = Conv2d(n, 32, (3, 3), (1, 1), act=None, padding='SAME', W_init=W_init, b_init=None, name='conv2')
            # n = BatchNormLayer(n, decay=0.999, act=None, is_train=is_train, name='conv2/bn')
            # n = PReluLayer(n, name='conv2/prelu')

        # net = _maxpool2d(net, 'pool1')
        # conv2
        # net = _conv2d(net, 128, (3, 3), (1, 1), None, 'SAME', 'conv2_1')
        # net = bn(net, 'bn2_1')
        # net = _conv2d(net, 128, (3, 3), (1, 1), None, 'SAME', 'conv2_2')
        # net = bn(net, 'bn2_2')
        for i in range(2):
            n = mobilenetv2_bottleneck(n, 64, filter_size=(3, 3), strides=((2, 2) if i == 0 else (1, 1)),\
                t=2, is_train=is_train, is_res=(False if i == 0 else True), data_format='channel_last', name='bottleneck1_{}'.format(i))

        # net = _maxpool2d(net, 'pool2')
        # conv3
        # net = _conv2d(net, 200, (3, 3), (1, 1), None, 'SAME', 'conv3_1')
        # net = bn(net, 'bn3_1')
        # net = _conv2d(net, 200, (3, 3), (1, 1), None, 'SAME', 'conv3_2')
        # net = bn(net, 'bn3_2')
        # net = _conv2d(net, 200, (3, 3), (1, 1), None, 'SAME', 'conv3_3')
        # net = bn(net, 'bn3_3')
        # # net = _conv2d(net, 200, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4') # VGG16 only 3 256
        for i in range(2):
            n = mobilenetv2_bottleneck(n, 128, filter_size=(3, 3), strides=((2, 2) if i == 0 else (1, 1)),\
                t=2, is_train=is_train, is_res=(False if i == 0 else True), data_format='channel_last', name='bottleneck2_{}'.format(i))

        # net = _maxpool2d(net, 'pool3')
        # # conv4
        # net = _conv2d(net, 384, (3, 3), (1, 1), None, 'SAME', 'conv4_1')
        # net = bn(net, 'bn4_1')
        # net = _conv2d(net, 384, (3, 3), (1, 1), None, 'SAME', 'conv4_2')
        for i in range(4):
            n = mobilenetv2_bottleneck(n, 256, filter_size=(3, 3), strides=((2, 2) if i == 0 else (1, 1)),\
                t=2, is_train=is_train, is_res=(False if i == 0 else True), data_format='channel_last', name='bottleneck3_{}'.format(i))

        # net = bn(net, 'bn4_2')
        net = _conv2d(n, 256, (3, 3), (1, 1), None, 'SAME', 'conv4_3')
        net = bn(net, 'bn4_3')
        net = _conv2d(net, 128, (3, 3), (1, 1), None, 'SAME', 'conv4_4')
        net = bn(net, 'bn4_4')
        return net

    b1_list = []
    b2_list = []
    with tf.variable_scope('model', reuse):
        # Feature extraction part
        # 1. by default, following the paper, we use VGG19 as the default model
        cnn = cnn_net(x, is_train=is_train)
        # cnn = tl.models.VGG19(x, end_with='conv4_2', reuse=reuse)
        # 2. you can customize this part to speed up the inferencing
        # cnn = tl.models.MobileNetV1(x, end_with='depth5', is_train=is_train, reuse=reuse)  # i.e. vgg16 conv4_2 ~ 4_4

        with tf.variable_scope("stage1/branch1"):
            b1 = _conv2d(cnn, 128, (3, 3), (1, 1), None, 'SAME', 'c1')
            b1 = bn(b1, 'bn1')
            b1 = _conv2d(b1, 128, (3, 3), (1, 1), None, 'SAME', 'c2')
            b1 = bn(b1, 'bn2')
            b1 = _conv2d(b1, 128, (3, 3), (1, 1), None, 'SAME', 'c3')
            b1 = bn(b1, 'bn3')
            # b1 = _conv2d(b1, 512, (1, 1), (1, 1), None, 'VALID', 'c4')
            b1 = _conv2d(b1, 128, (1, 1), (1, 1), None, 'VALID', 'c4')
            b1 = bn(b1, 'bn4')
            b1 = _conv2d_with_bias_init(b1, n_pos, (1, 1), (1, 1), None, 'VALID', 'confs')
            if is_train:
                b1.outputs = b1.outputs * mask_miss1
        with tf.variable_scope("stage1/branch2"):
            b2 = _conv2d(cnn, 128, (3, 3), (1, 1), None, 'SAME', 'c1')
            b2 = bn(b2, 'bn1')
            b2 = _conv2d(b2, 128, (3, 3), (1, 1), None, 'SAME', 'c2')
            b2 = bn(b2, 'bn2')
            b2 = _conv2d(b2, 128, (3, 3), (1, 1), None, 'SAME', 'c3')
            b2 = bn(b2, 'bn3')
            # b2 = _conv2d(b2, 512, (1, 1), (1, 1), None, 'VALID', 'c4')
            b2 = _conv2d(b2, 128, (1, 1), (1, 1), None, 'VALID', 'c4')
            b2 = bn(b2, 'bn4')
            b2 = _conv2d_with_bias_init(b2, 38, (1, 1), (1, 1), None, 'VALID', 'pafs')
            if is_train:
                b2.outputs = b2.outputs * mask_miss2
            b1_list.append(b1)
            b2_list.append(b2)

        # stage 2~6
        # for i in range(2, 7): # [2, 3, 4, 5, 6]
        for i in [5, 6]:
            b1, b2 = stage(cnn, b1_list[-1], b2_list[-1], n_pos, mask_miss1, mask_miss2, is_train, name='stage%d' % i)
            b1_list.append(b1)
            b2_list.append(b2)
        net = tl.layers.merge_networks([b1_list[-1], b2_list[-1]])
        return cnn, b1_list, b2_list, net


if __name__ == '__main__':
    x = tf.placeholder("float32", [None, 368, 368, 3])
    _, _, _, net = model(x, 19, None, None, False, False)
    net.print_layers()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tl.files.save_npz(net.all_params, 'model.npz', sess)
