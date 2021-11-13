import logging
from enum import Enum
import time
import functools
import multiprocessing
from distutils.dir_util import mkpath

import numpy as np
import tensorflow as tf
import cv2

from pycocotools.coco import maskUtils
from ..Config.define import MODEL,TRAIN,DATA,BACKBONE,KUNGFU,OPTIM

regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu

def read_imgfile(path, width, height, data_format='channels_last'):
    """Read image file and resize to network input size."""
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    val_image = val_image[:,:,::-1]
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    if data_format == 'channels_first':
        val_image = val_image.transpose([2, 0, 1])
    return val_image / 255.0


def get_sample_images(w, h):
    val_image = [
        read_imgfile('./images/p1.jpg', w, h),
        read_imgfile('./images/p2.jpg', w, h),
        read_imgfile('./images/p3.jpg', w, h),
        read_imgfile('./images/golf.jpg', w, h),
        read_imgfile('./images/hand1.jpg', w, h),
        read_imgfile('./images/hand2.jpg', w, h),
        read_imgfile('./images/apink1_crop.jpg', w, h),
        read_imgfile('./images/ski.jpg', w, h),
        read_imgfile('./images/apink2.jpg', w, h),
        read_imgfile('./images/apink3.jpg', w, h),
        read_imgfile('./images/handsup1.jpg', w, h),
        read_imgfile('./images/p3_dance.png', w, h),
    ]
    return val_image

def load_graph(model_file):
    """Load a freezed graph from file."""
    graph_def = tf.GraphDef()
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())

    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph

def get_op(graph, name):
    return graph.get_operation_by_name('import/%s' % name).outputs[0]


class Profiler(object):

    def __init__(self):
        self.count = dict()
        self.total = dict()

    def __del__(self):
        if self.count:
            self.report()

    def report(self):
        sorted_costs = sorted([(t, name) for name, t in self.total.items()])
        sorted_costs.reverse()
        names = [name for _, name in sorted_costs]
        hr = '-' * 80
        print(hr)
        print('%-12s %-12s %-12s %s' % ('tot (s)', 'count', 'mean (ms)', 'name'))
        print(hr)
        for name in names:
            tot, cnt = self.total[name], self.count[name]
            mean = tot / cnt
            print('%-12f %-12d %-12f %s' % (tot, cnt, mean * 1000, name))

    def __call__(self, name, duration):
        if name in self.count:
            self.count[name] += 1
            self.total[name] += duration
        else:
            self.count[name] = 1
            self.total[name] = duration


_default_profiler = Profiler()


def measure(f, name=None):
    if not name:
        name = f.__name__
    t0 = time.time()
    result = f()
    duration = time.time() - t0
    _default_profiler(name, duration)
    return result

def plot_humans(image, heatMat, pafMat, humans, name):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    a = fig.add_subplot(2, 3, 1)

    plt.imshow(draw_humans(image, humans))

    a = fig.add_subplot(2, 3, 2)
    tmp = np.amax(heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 3, 4)
    a.set_title('Vectormap-x')
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 3, 5)
    a.set_title('Vectormap-y')
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    mkpath('vis')
    plt.savefig('vis/result-%s.png' % name)

def tf_repeat(tensor, repeats):
    """
    Args:

    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input

    Returns:

    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    expanded_tensor = tf.expand_dims(tensor, -1)
    multiples = [1] + repeats
    tiled_tensor = tf.tile(expanded_tensor, multiples=multiples)
    repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor

def decode_mask(meta_mask_list):
    if(type(meta_mask_list)!=list):
        return None
    if(meta_mask_list==[]):
        return None
    inv_mask_list=[]
    for meta_mask in meta_mask_list:
        mask=maskUtils.decode(meta_mask)
        inv_mask=np.logical_not(mask)
        inv_mask_list.append(inv_mask)
    mask=np.ones_like(inv_mask_list[0])
    for inv_mask in inv_mask_list:
        mask=np.logical_and(mask,inv_mask)
    mask = mask.astype(np.uint8)
    return mask

def regulize_loss(target_model,weight_decay_factor):
    re_loss=0
    regularizer=tf.keras.regularizers.l2(l=weight_decay_factor)
    for trainable_weight in target_model.trainable_weights:
        re_loss+=regularizer(trainable_weight)
    return re_loss

def pad_image(img,stride,pad_value=0.0):
    img_h,img_w,img_c=img.shape
    pad_h= 0 if (img_h%stride==0) else int(stride-(img_h%stride))
    pad_w= 0 if (img_w%stride==0) else int(stride-(img_w%stride))
    pad=[pad_h//2,pad_h-pad_h//2,pad_w//2,pad_w-pad_w//2]
    padded_image=np.zeros(shape=(img_h+pad_h,img_w+pad_w,img_c))+pad_value
    padded_image[pad[0]:img_h+pad[0],pad[2]:img_w+pad[2],:]=img
    return padded_image,pad

def pad_image_shape(img,shape,pad_value=0.0):
    img_h,img_w,img_c=img.shape
    dst_h,dst_w=shape
    pad_h=dst_h-img_h
    pad_w=dst_w-img_w
    pad=[pad_h//2,pad_h-pad_h//2,pad_w//2,pad_w-pad_w//2]
    padded_image=np.zeros(shape=(img_h+pad_h,img_w+pad_w,img_c))+pad_value
    padded_image[pad[0]:img_h+pad[0],pad[2]:img_w+pad[2],:]=img
    return padded_image,pad

def scale_image(image,hin,win,scale_rate=0.95):
    #scale a image into the size of scale_rate*hin and scale_rate*win
    #used for model inferecne
    image_h,image_w,_=image.shape
    scale_h,scale_w=int(scale_rate*image_h),int(scale_rate*image_w)
    scale_image=cv2.resize(image,(scale_w,scale_h),interpolation=cv2.INTER_CUBIC)
    padded_image,pad=pad_image_shape(scale_image,shape=(hin,win),pad_value=0.0)
    return padded_image,pad

def get_optim(optim_type):
    if(optim_type==OPTIM.Adam):
        print("using optimizer Adam!")
        return tf.keras.optimizers.Adam
    elif(optim_type==OPTIM.RMSprop):
        print("using optimizer RMSProp!")
        return tf.keras.optimizers.RMSprop
    elif(optim_type==OPTIM.SGD):
        print("using optimizer SGD!")
        return tf.keras.optimizers.SGD
    else:
        raise NotImplementedError("invalid optim type")

def regulize_loss(target_model,weight_decay_factor):
    re_loss=0
    regularizer=tf.keras.regularizers.l2(l=weight_decay_factor)
    for weight in target_model.trainable_weights:
        re_loss+=regularizer(weight)
    return re_loss

def resize_CHW(x, dst_shape):
    x = x[np.newaxis,:,:,:]
    x = resize_NCHW(x, dst_shape)
    x = x[0]
    return x

def resize_NCHW(x, dst_shape):
    x = tf.transpose(x,[0,2,3,1])
    x = tf.image.resize(x, dst_shape)
    x = tf.transpose(x,[0,3,1,2])
    return x

def NCHW_to_NHWC(x):
    return tf.transpose(x,[0,2,3,1])

def NHWC_to_NCHW(x):
    return tf.transpose(x,[0,3,1,2])

def to_tensor_dict(dict_x):
    for key in dict_x.keys():
        dict_x[key]=tf.convert_to_tensor(dict_x[key])
    return dict_x

def to_numpy_dict(dict_x):
    for key in dict_x.keys():
        value=dict_x[key]
        if(type(value) is not np.ndarray):
            value=value.numpy()
        dict_x[key]=value
    return dict_x

def get_num_parallel_calls():
    return max(multiprocessing.cpu_count()//2,1)

@functools.lru_cache(maxsize=16)
def get_meshgrid(mesh_h,mesh_w):
    x_range=np.linspace(start=0,stop=mesh_w-1,num=mesh_w)
    y_range=np.linspace(start=0,stop=mesh_h-1,num=mesh_h)
    mesh_x,mesh_y=np.meshgrid(x_range,y_range)
    mesh_grid=np.stack([mesh_x,mesh_y])
    return mesh_grid

def log_model(msg):
    logger=logging.getLogger("MODEL")
    logger.info(msg)

def log_train(msg):
    logger=logging.getLogger("TRAIN")
    logger.info(msg)

def image_float_to_uint8(image):
    return np.clip(image*255,0,255).astype(np.uint8)