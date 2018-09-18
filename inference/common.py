from enum import Enum
import time
from distutils.dir_util import mkpath

import numpy as np
import tensorflow as tf
import cv2

regularizer_conv = 0.004
regularizer_dsconv = 0.0004
batchnorm_fused = True
activation_fn = tf.nn.relu


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


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


CocoPairs = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
             (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)]  # = 19
CocoPairsRender = CocoPairs[:-2]
# CocoPairsNetwork = [
#     (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
#     (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
#  ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


def read_imgfile(path, width=None, height=None, data_format='channels_last'):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    if data_format == 'channels_first':
        val_image = val_image.transpose([2, 0, 1])
    return val_image


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
        pass
        # TODO: make sure __def__ is only called on exit
        # self.report()

    def report(self):
        sorted_costs = sorted([(t, name) for name, t in self.total.items()])
        sorted_costs.reverse()
        names = [name for _, name in sorted_costs]
        print('%-12s %-12s %-12s %s' % ('tot (s)', 'count', 'mean (s)', 'name'))
        for name in names:
            tot, cnt = self.total[name], self.count[name]
            mean = tot / cnt
            print('%-12f %-12d %-12f %s' % (tot, cnt, mean, name))

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
    # print('start %s' % name)
    duration = time.time() - t0
    _default_profiler(name, duration)
    # line = '%s took %fs' % (name, duration)
    # print(line)
    # with open('profile.log', 'a') as f:
    #     f.write(line + '\n')
    return result


def plot_humans(e, image, humans, name):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    a = fig.add_subplot(2, 3, 1)

    plt.imshow(e.draw_humans(image[..., ::-1], humans, True))

    a = fig.add_subplot(2, 3, 2)
    # plt.imshow(cv2.resize(image, (e.heatMat.shape[1], e.heatMat.shape[0])), alpha=0.5)
    tmp = np.amax(e.heatMat[:, :, :-1], axis=2)
    plt.imshow(tmp, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    tmp2 = e.pafMat.transpose((2, 0, 1))
    tmp2_odd = np.amax(np.absolute(tmp2[::2, :, :]), axis=0)
    tmp2_even = np.amax(np.absolute(tmp2[1::2, :, :]), axis=0)

    a = fig.add_subplot(2, 3, 4)
    a.set_title('Vectormap-x')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_odd, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()

    a = fig.add_subplot(2, 3, 5)
    a.set_title('Vectormap-y')
    # plt.imshow(CocoPose.get_bgimg(inp, target_size=(vectmap.shape[1], vectmap.shape[0])), alpha=0.5)
    plt.imshow(tmp2_even, cmap=plt.cm.gray, alpha=0.5)
    plt.colorbar()
    mkpath('vis')
    plt.savefig('vis/result-%s.png' % name)
