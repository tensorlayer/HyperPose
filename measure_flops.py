import argparse
from argparse import ArgumentParser
import tensorflow as tf
# notes:
# now, 'savemodel' format in tensorflow2, 'pb' format in tensorflow1, 'onnx' format in pytorch are
# official inference formats for each platform, current this script only suppory 'pb' format calculation
print(f"This script currently only support calculation flops of pb format model!")
argparser=ArgumentParser()
argparser.add_argument("--model_path",type=str,default=f"./save_dir/default_model/newest_model.pb",\
    help="the path to the model file")
args=argparser.parse_args()

#load graph_def from pb format graph_file
model_path=args.model_path
graph_file=tf.io.gfile.GFile(model_path,"rb")
graph_def=tf.compat.v1.GraphDef()
graph_def.ParseFromString(graph_file.read())
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def)    
    run_meta=tf.compat.v1.RunMetadata()
    options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    flops=tf.compat.v1.profiler.profile(graph,run_meta=run_meta,options=options)
    print(f"model: {model_path} GFLOPS: {flops.total_float_ops/1e9}")