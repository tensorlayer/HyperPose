import pathlib
import tensorflow as tf
from functools import partial
from hyperpose import Config,Model,Dataset

#load model weights from hyperpose
Config.set_model_name("new_pifpaf")
Config.set_model_type(Config.MODEL.Pifpaf)
Config.set_dataset_type(Config.DATA.MSCOCO)
config=Config.get_config()
model=Model.get_model(config)
model.load_weights(f"{config.model.model_dir}/newest_model.npz")
model.eval()
#construct representative dataset used for quantization(here using the first 100 validate images)
scale_image_func=partial(Model.common.scale_image,hin=model.hin,win=model.win,scale_rate=0.95)
def decode_image(image_file,image_id):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    scaled_image,pad = tf.py_function(scale_image_func,[image],[tf.float32,tf.float32])
    return scaled_image
dataset=Dataset.get_dataset(config)
val_dataset=dataset.get_eval_dataset()
rep_dataset=val_dataset.enumerate()
rep_dataset=rep_dataset.filter(lambda i,image_data : i<=100)
rep_dataset=rep_dataset.map(lambda i,image_data: image_data)
rep_dataset=rep_dataset.map(decode_image).batch(1)
print(f"test rep_dataset:{rep_dataset}")
#covert to tf-lite using int8-only quantization
input_signature=tf.TensorSpec(shape=(None,3,None,None))
converter=tf.lite.TFLiteConverter.from_concrete_functions([model.infer.get_concrete_function(x=input_signature)])
converter.optimizations=[tf.lite.Optimize.DEFAULT]
converter.representative_dataset=rep_dataset
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()
print("model quantized using uint8 quantization!")
#save the converted quantization model
save_path=f"./save_dir/{config.model.model_name}.tflite"
tf.io.write_file(save_path,tflite_model_quant)
#print(f"export tflite file finished! output file: {save_path}")




