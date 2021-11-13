#!/usr/bin/env python3
import os
import cv2
import sys
import glob 
import argparse
import matplotlib
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config,Model,Dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperpose')
    parser.add_argument("--model_type",
                        type=str,
                        default="Openpose",
                        help="human pose estimation model type, available options: Openpose, LightweightOpenpose ,MobilenetThinOpenpose, PoseProposal, Pifpaf")
    parser.add_argument("--model_backbone",
                        type=str,
                        default="Default",
                        help="model backbone, available options: Mobilenet, Vggtiny, Vgg19, Resnet18, Resnet50")
    parser.add_argument("--dataset_type",
                        type=str,
                        default="MSCOCO",
                        help="dataset name,to determine which dataset to use, available options: MSCOCO, MPII ")
    parser.add_argument("--model_name",
                        type=str,
                        default="default_name",
                        help="model name,to distinguish model and determine model dir")
    parser.add_argument("--image_dir",
                        type=str,
                        default="./save_dir/example_dir/image",
                        help="image paths to be processed by the model"
                        )
    parser.add_argument("--output_dir",
                        type=str,
                        default="./save_dir/example_dir/output_dir",
                        help="ouput directory of the model forwarding"
                        )

args=parser.parse_args()
# config model
Config.set_model_name(args.model_name)
Config.set_model_type(Config.MODEL[args.model_type])
Config.set_dataset_type(Config.DATA[args.dataset_type])
Config.set_model_backbone(Config.BACKBONE[args.model_backbone])
config = Config.get_config()
output_dir = os.path.join(args.output_dir,args.model_name)
os.makedirs(output_dir, exist_ok=True)

# contruct model and processors
model = Model.get_model(config)
# visualizer
VisualizerClass = Model.get_visualizer(config)
visualizer = VisualizerClass(save_dir=output_dir, parts=model.parts, limbs=model.limbs)
# post processor
PostProcessorClass = Model.get_postprocessor(config)
post_processor = PostProcessorClass(parts=model.parts, limbs=model.limbs, hin=model.hin, win=model.win, hout=model.hout,
                                    wout=model.wout, colors=model.colors)
# image processor
ImageProcessorClass = Model.get_imageprocessor()
image_processor = ImageProcessorClass(input_h=model.hin, input_w=model.win)

# load weights
model_weight_path = f"./save_dir/{args.model_name}/model_dir/newest_model.npz"
model.load_weights(model_weight_path, format="npz_dict")
model.eval()
# begin process
for image_path in glob.glob(f"{args.image_dir}/*"):
    image_name = os.path.basename(image_path)
    print(f"processing image:{image_name}")
    # image read, normalize, and scale
    image = image_processor.read_image_rgb_float(image_path)
    input_image, scale, pad = image_processor.image_pad_and_scale(image)
    input_image = np.transpose(input_image,[2,0,1])[np.newaxis,:,:,:]
    # model forward
    predict_x = model.forward(input_image)
    # post process
    humans = post_processor.process(predict_x)[0]
    # visualize heatmaps
    visualizer.visualize(image_batch=input_image, predict_x=predict_x, humans_list=[humans], name=image_name)
    # visualize results (restore detected humans)
    print(f"{len(humans)} humans detected")
    for human_idx,human in enumerate(humans,start=1):
        human.unpad(pad)
        human.unscale(scale)
        print(f"human:{human_idx} num of detected body joints:{human.get_partnum()}")
        human.print()
    visualizer.visualize_result(image=image, humans=humans, name=f"{image_name}_result")

    
    

