import os
from .define import MODEL,DATA,TRAIN,BACKBONE
from easydict import EasyDict as edict

#model configuration
model = edict()
 # number of keypoints, including instance
model.K_size = 18 
model.L_size = 17
# input size during training
model.hin = 384  
model.win = 384
# output size during training
model.hout = 12 
model.wout = 12
#neibor size during training
model.hnei=9
model.wnei=9
#loss weights
model.lmd_rsp=0.25
model.lmd_iou=1
model.lmd_coor=5.0
model.lmd_size=5.0
model.lmd_limb=0.5
model.model_type = MODEL.PoseProposal 
model.model_name = "default_name"
model.model_backbone=BACKBONE.Default
model.data_format = "channels_first"
#save directory
model.model_dir= f"./save_dir/{model.model_name}/model_dir" 

#train configuration
train = edict()
train.batch_size = 22
train.save_interval = 5000
train.n_step = 1040000
train.lr_init = 1e-4  # initial learning rate
train.lr_decay_factor=0.9
train.weight_decay_factor = 5e-4
train.train_type=TRAIN.Single_train
train.vis_dir = f"./save_dir/{model.model_name}/train_vis_dir"

#eval configuration
eval =edict()
eval.batch_size=22
eval.vis_dir= f"./save_dir/{model.model_name}/eval_vis_dir"

#test configuration
test =edict()
test.vis_dir=f"./save_dir/{model.model_name}/test_vis_dir"

#data configuration
data = edict()
data.dataset_type = DATA.MSCOCO
data.dataset_version = "2017" 
data.dataset_path = "./data"
data.dataset_filter=None
data.vis_dir=f"./save_dir/data_vis_dir"

#log configuration
log = edict()
log.log_interval = 1
log.log_path = f"./save_dir/{model.model_name}/log.txt"