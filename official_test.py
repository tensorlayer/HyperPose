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
from hyperpose import Config,Model,Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FastPose.')
    parser.add_argument("--model_type",
                        type=str,
                        default="Openpose",
                        help="human pose estimation model type, available options: Openpose, LightweightOpenpose ,PoseProposal")
    parser.add_argument("--model_backbone",
                        type=str,
                        default="Default",
                        help="model backbone, available options: Mobilenet, Vgg19, Resnet18, Resnet50")
    parser.add_argument("--model_name",
                        type=str,
                        default="default_name",
                        help="model name,to distinguish model and determine model dir")
    parser.add_argument("--dataset_type",
                        type=str,
                        default="MSCOCO",
                        help="dataset name,to determine which dataset to use, available options: coco ")
    parser.add_argument("--dataset_version",
                        type=str,
                        default="2017",
                        help="dataset version, only use for MSCOCO and available for version 2014 and 2017 ")
    parser.add_argument("--dataset_path",
                        type=str,
                        default="data",
                        help="dataset path,to determine the path to load the dataset")
    parser.add_argument('--train_type',
                        type=str,
                        default="Single_train",
                        help='train type, available options: Single_train, Parallel_train')
    parser.add_argument('--kf_optimizer',
                        type=str,
                        default='Pair_avg',
                        help='kung fu parallel optimizor,available options: Sync_sgd, Sync_avg, Pair_avg')
    parser.add_argument('--test_num',
                        type=int,
                        default=100000,
                        help='number of test')
    parser.add_argument('--vis_num',
                        type=int,
                        default=60,
                        help='number of visible test')
    parser.add_argument('--multiscale',
                        type=bool,
                        default=False,
                        help='enable multiscale_search')
                        

    args=parser.parse_args()
    Config.set_model_name(args.model_name)
    Config.set_model_type(Config.MODEL[args.model_type])
    Config.set_model_backbone(Config.BACKBONE[args.model_backbone])
    Config.set_dataset_type(Config.DATA[args.dataset_type])
    Config.set_dataset_path(args.dataset_path)
    Config.set_dataset_version(args.dataset_version)
    
    config=Config.get_config()
    model=Model.get_model(config)
    test=Model.get_test(config)
    dataset=Dataset.get_dataset(config)

    test(model,dataset,vis_num=args.vis_num,total_test_num=args.test_num,enable_multiscale_search=args.multiscale)
