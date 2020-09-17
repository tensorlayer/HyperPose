from easydict import EasyDict as edict

pretrain=edict()
pretrain.enable=False
pretrain.lr_init=5e-4
pretrain.batch_size=32
pretrain.total_step=370000000
pretrain.log_interval=100
pretrain.val_interval=5000
pretrain.save_interval=5000
pretrain.weight_decay_factor=1e-5
pretrain.pretrain_dataset_path="./data/imagenet"
pretrain.pretrain_model_dir="./save_dir/pretrain_backbone"
pretrain.val_num=20000
pretrain.lr_decay_step=170000
