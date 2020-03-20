#!/usr/bin/env python3

import os
import cv2
import sys
import math
import json
import time
import argparse
import matplotlib
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
sys.path.append('.')
matplotlib.use('Agg')

from train_configs import init_config
from models import get_model,get_train
from datasets import get_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastPose.')
    parser.add_argument("--model_type",
                        type=str,
                        default="lightweight_openpose",
                        help="human pose estimation model type, available options: lightweight_openpose, pose_proposal")
    parser.add_argument("--model_name",
                        type=str,
                        default="default_name",
                        help="model name,to distinguish model and determine model dir")
    parser.add_argument("--dataset_name",
                        type=str,
                        default="coco",
                        help="dataset name,to determine which dataset to use, available options: coco ")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="data",
                        help="dataset path,to determine the path to load the dataset")
    parser.add_argument('--parallel',
                        action='store_true',
                        default=False,
                        help='enable parallel training')
    parser.add_argument('--kf-optimizer',
                        type=str,
                        default='sma',
                        help='kung fu parallel optimizor,available options: sync-sgd, async-sgd, sma')
                        

    args=parser.parse_args()
    config=init_config(model_type=args.model_type,model_name=args.model_name,dataset_path=args.dataset_path)
    train_model=get_model(args.model_type,config)
    train_dataset=get_dataset(args.dataset_name,config)
    single_train,parallel_train=get_train(args.model_type)

    if args.parallel:
        parallel_train(train_model,train_dataset,config,args.kf_optimizer)
    else:
        single_train(train_model,train_dataset,config)
