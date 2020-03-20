#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf
import tensorlayer as tl
from train_configs import init_config
from models import get_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export fastpose models to pb format.')
    parser.add_argument("--model_type",
                        type=str,
                        default="lightweight_openpose",
                        help="human pose estimation model type, available options: lightweight_openpose, pose_proposal")
    parser.add_argument("--model_name",
                        type=str,
                        default="default_name",
                        help="model name,to distinguish model and determine model directory")
    parser.add_argument("--output_dir",
                        type=str,
                        default="save_dir",
                        help="which dir to output the exported pb model")
    
    args=parser.parse_args()    
    config=init_config(model_type=args.model_type,model_name=args.model_name)
    export_model=get_model(args.model_type,config)
    input_path=f"{config.MODEL.model_dir}/newest_model.npz"
    output_dir=f"{args.output_dir}/{config.MODEL.model_name}"
    output_path=f"{output_dir}/frozen_{config.MODEL.model_name}.pb"
    print(f"exporting model {config.MODEL.model_name} from {input_path}...")
    if(not os.path.exists(output_dir)):
        print("creating output_dir...")
        os.mkdir(output_dir)
    if(not os.path.exists(input_path)):
        print("input model file doesn't exist!")
        print("conversion aborted!")
    else:
        export_model.load_weights(input_path)
        export_model.eval()
        input_signature=tf.TensorSpec(shape=(None,export_model.hin,export_model.hout,3))
        concrete_function=export_model.forward.get_concrete_function(x=input_signature)
        frozen_graph=convert_variables_to_constants_v2(concrete_function)
        frozen_graph_def=frozen_graph.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_graph_def,logdir=output_dir,name=f"frozen_{args.model_name}.pb",\
            as_text=False)
        print(f"exporting pb file finished! output file: {output_path}")

