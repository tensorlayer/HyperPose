import os

from easydict import EasyDict as edict

config = edict()

config.TRAIN = edict()
config.TRAIN.batch_size = 8
config.TRAIN.save_interval = 5000
config.TRAIN.log_interval = 1
# config.TRAIN.n_epoch = 80
config.TRAIN.n_step = 600000  # total number of step
config.TRAIN.lr_init = 4e-5  # initial learning rate
config.TRAIN.lr_decay_every_step = 136106  # evey number of step to decay lr
config.TRAIN.lr_decay_factor = 0.333  # decay lr factor
config.TRAIN.weight_decay_factor = 5e-4

config.MODEL = edict()
config.MODEL.model_path = 'models'  # save directory
config.MODEL.n_pos = 19  # number of keypoints + 1 for background
config.MODEL.hin = 368  # input size during training , 240
config.MODEL.win = 368
config.MODEL.hout = int(config.MODEL.hin / 8)  # output size during training (default 46)
config.MODEL.wout = int(config.MODEL.win / 8)
config.MODEL.name = 'vgg'  # vgg, vggtiny, mobilenet

if (config.MODEL.hin % 16 != 0) or (config.MODEL.win % 16 != 0):
    raise Exception("image size should be divided by 16")

config.DATA = edict()
config.DATA.train_data = 'coco'  # coco, custom, coco_and_custom
config.DATA.coco_version = '2017'  # MSCOCO version 2014 or 2017
config.DATA.data_path = 'data'
config.DATA.your_images_path = os.path.join('data', 'your_data', 'images')
config.DATA.your_annos_path = os.path.join('data', 'your_data', 'coco.json')

config.LOG = edict()
config.LOG.vis_path = 'vis'

# config.VALID = edict()

# import json
# def log_config(filename, cfg):
#     with open(filename, 'w') as f:
#         f.write("================================================\n")
#         f.write(json.dumps(cfg, indent=4))
#         f.write("\n================================================\n")
