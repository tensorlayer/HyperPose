#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config,Model,Dataset
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export fastpose models to pb format.')
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
                        default='Sma',
                        help='kung fu parallel optimizor,available options: Sync_sgd, Async_sgd, Sma')
    parser.add_argument("--output_dir",
                        type=str,
                        default="save_dir",
                        help="which dir to output the exported pb model")
    
    
    args=parser.parse_args()    
    Config.set_model_name(args.model_name)
    Config.set_model_type(Config.MODEL[args.model_type])
    Config.set_model_backbone(Config.BACKBONE[args.model_backbone])
    config=Config.get_config()
    export_model=Model.get_model(config)

    input_path=f"{config.model.model_dir}/newest_model.npz"
    output_dir=f"{args.output_dir}/{config.model.model_name}"
    output_path=f"{output_dir}/frozen_{config.model.model_name}.pb"
    print(f"exporting model {config.model.model_name} from {input_path}...")
    if(not os.path.exists(output_dir)):
        print("creating output_dir...")
        os.mkdir(output_dir)
    if(not os.path.exists(input_path)):
        print("input model file doesn't exist!")
        print("conversion aborted!")
    else:
        export_model.load_weights(input_path)
        export_model.eval()
        if(export_model.data_format=="channels_last"):
            input_signature=tf.TensorSpec(shape=(None,None,None,3))
        else:
            input_signature=tf.TensorSpec(shape=(None,3,None,None))
        concrete_function=export_model.infer.get_concrete_function(x=input_signature)
        frozen_graph=convert_variables_to_constants_v2(concrete_function)
        frozen_graph_def=frozen_graph.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_graph_def,logdir=output_dir,name=f"frozen_{args.model_name}.pb",\
            as_text=False)
        print(f"exporting pb file finished! output file: {output_path}")

