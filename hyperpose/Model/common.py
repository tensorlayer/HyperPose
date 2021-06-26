import logging
from enum import Enum
import time
from distutils.dir_util import mkpath

import numpy as np
import tensorflow as tf
import cv2
from ..Config.define import MODEL,TRAIN,DATA,BACKBONE,KUNGFU,OPTIM

regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu

class MPIIPart(Enum):
    RAnkle = 0
    RKnee = 1
    RHip = 2
    LHip = 3
    LKnee = 4
    LAnkle = 5
    RWrist = 6
    RElbow = 7
    RShoulder = 8
    LShoulder = 9
    LElbow = 10
    LWrist = 11
    Neck = 12
    Head = 13

    @staticmethod
    def from_coco(human):
        # t = {
        #     MPIIPart.RAnkle: CocoPart.RAnkle,
        #     MPIIPart.RKnee: CocoPart.RKnee,
        #     MPIIPart.RHip: CocoPart.RHip,
        #     MPIIPart.LHip: CocoPart.LHip,
        #     MPIIPart.LKnee: CocoPart.LKnee,
        #     MPIIPart.LAnkle: CocoPart.LAnkle,
        #     MPIIPart.RWrist: CocoPart.RWrist,
        #     MPIIPart.RElbow: CocoPart.RElbow,
        #     MPIIPart.RShoulder: CocoPart.RShoulder,
        #     MPIIPart.LShoulder: CocoPart.LShoulder,
        #     MPIIPart.LElbow: CocoPart.LElbow,
        #     MPIIPart.LWrist: CocoPart.LWrist,
        #     MPIIPart.Neck: CocoPart.Neck,
        #     MPIIPart.Nose: CocoPart.Nose,
        # }
        
        from ..Dataset.mscoco_dataset.define import CocoPart
        t = [
            (MPIIPart.Head, CocoPart.Nose),
            (MPIIPart.Neck, CocoPart.Neck),
            (MPIIPart.RShoulder, CocoPart.RShoulder),
            (MPIIPart.RElbow, CocoPart.RElbow),
            (MPIIPart.RWrist, CocoPart.RWrist),
            (MPIIPart.LShoulder, CocoPart.LShoulder),
            (MPIIPart.LElbow, CocoPart.LElbow),
            (MPIIPart.LWrist, CocoPart.LWrist),
            (MPIIPart.RHip, CocoPart.RHip),
            (MPIIPart.RKnee, CocoPart.RKnee),
            (MPIIPart.RAnkle, CocoPart.RAnkle),
            (MPIIPart.LHip, CocoPart.LHip),
            (MPIIPart.LKnee, CocoPart.LKnee),
            (MPIIPart.LAnkle, CocoPart.LAnkle),
        ]

        pose_2d_mpii = []
        visibilty = []
        for mpi, coco in t:
            if coco.value not in human.body_parts.keys():
                pose_2d_mpii.append((0, 0))
                visibilty.append(False)
                continue
            pose_2d_mpii.append((human.body_parts[coco.value].x, human.body_parts[coco.value].y))
            visibilty.append(True)
        return pose_2d_mpii, visibilty


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


def draw_humans(npimg, humans):
    npimg = np.copy(npimg)
    image_h, image_w = npimg.shape[:2]
    centers = {}
    for human in humans:
        # draw point
        for i in range(CocoPart.Background.value):
            if i not in human.body_parts.keys():
                continue

            body_part = human.body_parts[i]
            center = (int(body_part.x * image_w + 0.5), int(body_part.y * image_h + 0.5))
            centers[i] = center
            cv2.circle(npimg, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in human.body_parts.keys() or pair[1] not in human.body_parts.keys():
                continue
            cv2.line(npimg, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return npimg


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


def rename_tensor(x, name):
    # FIXME: use tf.identity(x, name=name) doesn't work
    new_shape = []
    for d in x.shape:
        try:
            d = int(d)
        except:
            d = -1
        new_shape.append(d)
    return tf.reshape(x, new_shape, name=name)

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

def init_log(config):
    logger=logging.getLogger(name="hyperpose")
    file_handler=logging.FileHandler(filename=config.log.log_path,mode="a")
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

def log(msg):
    logger=logging.getLogger(name="hyperpose")
    logger.info(msg=msg)
    print(msg)
