#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf
from hyperpose import Config, Model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def analyze_inputs_outputs(graph):
    ops = graph.get_operations()
    outputs_set = set(ops)
    inputs = []
    for op in ops:
        if len(op.inputs) == 0 and op.type == 'Placeholder':
            inputs.append(op)
        else:
            for input_tensor in op.inputs:
                if input_tensor.op in outputs_set:
                    outputs_set.remove(input_tensor.op)
    outputs = list(outputs_set)
    # Control nodes shall not be considered.
    # input like: "import/x" -> x
    # output like: "import/Identity", "import/Identity_1" -> Identity, Identity_1
    inputs = [x.name.split('/')[-1] for x in inputs if '_control_node' not in x.name]
    outputs = [x.name.split('/')[-1] for x in outputs if '_control_node' not in x.name]
    return (inputs, outputs)

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
    parser.add_argument("--output_dir",
                        type=str,
                        default="save_dir",
                        help="which dir to output the exported pb model")
    parser.add_argument("--export_batch_size",
                        type=int,
                        default=None,
                        help="the expected input image batch_size of the converted model, set to None to support dynamic shape"
                        )
    parser.add_argument("--export_h",
                        type=int,
                        default=None,
                        help="the expected input image height of the converted model, set to None to support dynamic shape"
                        )
    parser.add_argument("--export_w",
                        type=int,
                        default=None,
                        help="the expected input image width of the converted model, set to None to support dynamic shape")
                        
    
    args=parser.parse_args()    
    Config.set_model_name(args.model_name)
    Config.set_model_type(Config.MODEL[args.model_type])
    Config.set_model_backbone(Config.BACKBONE[args.model_backbone])
    config=Config.get_config()
    export_model=Model.get_model(config)

    export_batch_size=args.export_batch_size
    export_h,export_w=args.export_h,args.export_w
    print(f"export_batch_size={export_batch_size}\texport_h={export_h}\texport_w={export_w}")
    input_path=f"{config.model.model_dir}/newest_model.npz"
    output_dir=f"{args.output_dir}/{config.model.model_name}"
    output_path=f"{output_dir}/frozen_{config.model.model_name}.pb"
    print(f"Exporting model {config.model.model_name} from {input_path}...")
    if(not os.path.exists(output_dir)):
        print("Creating output_dir...")
        os.mkdir(output_dir)
    if(not os.path.exists(input_path)):
        print("Input model file doesn't exist!")
        print("Conversion aborted!")
    else:
        export_model.load_weights(input_path)
        export_model.eval()
        if(export_model.data_format=="channels_last"):
            input_signature=tf.TensorSpec(shape=(export_batch_size,export_h,export_w,3))
        else:
            input_signature=tf.TensorSpec(shape=(export_batch_size,3,export_h,export_w))
        concrete_function=export_model.infer.get_concrete_function(x=input_signature)
        frozen_graph=convert_variables_to_constants_v2(concrete_function)
        frozen_graph_def=frozen_graph.graph.as_graph_def()
        tf.io.write_graph(graph_or_graph_def=frozen_graph_def,logdir=output_dir,name=f"frozen_{args.model_name}.pb",\
            as_text=False)
        print(f"Exporting pb file finished! output file: {output_path}")

        with tf.Graph().as_default()  as graph:
            tf.import_graph_def(frozen_graph_def)
            input_names, output_names = analyze_inputs_outputs(graph)
            print(f'Exported graph INPUT nodes: {input_names}')
            print(f'Exported graph OUTPUT nodes: {output_names}')
