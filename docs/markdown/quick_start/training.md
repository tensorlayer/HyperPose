# Quick Start of Training Library

## Prerequisites
* Make sure you have configured 'hyperpose' virtual environment following the training installation guide,(if not, you can refer to [training installation](../install/training.md)).
* Make sure your GPU is available now(using tf.test.is_gpu_available() and it should return True)
* Make sure the Hyperpose training Library is under the root directory of the project(where you write train.py and eval.py)

## Train a model
The training procedure of Hyperpose is to set the model architecture, model backbone and dataset.
User specify these configuration using the seting functions of Config module with predefined enum value.
The code for training as simple as following would work.
```bash
# >>> import modules of hyperpose
from hyperpose import Config,Model,Dataset
# >>> set model name is necessary to distinguish models (neccesarry)
Config.set_model_name(args.model_name)
# >>> set model architecture(and model backbone when in need)
Config.set_model_type(Config.MODEL.LightweightOpenpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# >>> set dataset to use
Config.set_dataset_type(Config.DATA.COCO)
# >>> set training type 
Config.set_train_type(Config.TRAIN.Single_train)
# >>> configuration is done, get config object to assemble the system
config=Config.get_config()
model=Model.get_model(config)
dataset=Dataset.get_dataset(config)
train=Model.get_train(config)
# >>> train!
train(model,dataset)
```
Then the integrated training pipeline will start.
for each model, Hyperpose will save all the related files in the direatory:
./save_dir/model_name, where *model_name* is the name user set by using *Config.set_model_name*
the directory and its contents are below:  
* directory to save model                      ./save_dir/model_name/model_dir  
* directory to save train result               ./save_dir/model_name/train_vis_dir  
* directory to save evaluate result            ./save_dir/model_name/eval_vis_dir  
* directory to save dataset visualize result   ./save_dir/model_name/data_vis_dir  
* file path to save train log                  ./save_dir/model_name/log.txt  

The above code section show the simplest way to use Hyperpose training library, to make full use of Hyperpose training library,
you can refer to [training tutorial](../tutorial/training.md)

## Eval a model
The evaluate procedure using Hyperpose is almost the same to the training procedure:
the model will be loaded from the ./save_dir/model_name/model_dir/newest_model.npz
```bash
# >>> import modules of hyperpose
from hyperpose import Config,Model,Dataset
# >>> set model name to be eval
Config.set_model_name(args.model_name)
# >>> the model architecture and backbone setting should be the same with the training configuration of the model to be evaluated.
Config.set_model_type(Config.MODEL.LightweightOpenpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# >>> set dataset to use
Config.set_dataset_type(Config.DATA.COCO)
# >>> configuration is done, get config object to assemble the system
config=Config.get_config()
model=Model.get_model(config)
dataset=Dataset.get_dataset(config)
eval=Model.get_eval(config)
# >>> eval
eval(model,dataset)
```
Then the integrated evaluation pipeline will start, the final evaluate metrics will be output at last.
It should be noted that:
1.the model architecture, model backbone, dataset type should be the same with the configuration under which model was trained.
2.the evaluation metrics will follow the official evaluation metrics of dataset

The above code section show the simplest way to use Hyperpose training library to evaluate a model trained by Hyperpose, to make full use of Hyperpose training library, you can refer to [training tutorial](../tutorial/training.md)

## Export a model
To export a model trained by Hyperpose, one should follow two step:
* (1)convert the trained .npz model into .pb format
    this can be done either call the export_pb.py from Hyperpose repo
    ```bash
    python export_pb.py --model_type=your_model_type --model_name=your_model_name
    ```
    then the converted model will be put in the ./save_dir/model_name/forzen_model_name.pb
    one can also export himself by loading model and using get_concrete_function by himself, please refer the tutorial for details
* (2)convert the frozen .pb format model by tensorflow-onnx
    Make sure you have installed the extra requirements for exporting models from [training installation](../install/training.md)<br>
    if you don't know the input and output name of the pb model,you should use the function *summarize_graph* function 
    of graph_transforms from tensorflow
    ```bash
    bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=your_frozen_model.pb
    ```
    then, after knowing the input and output nodes of your .pb model,use tf2onnx
    ```bash
    python -m tf2onnx.convert --graphdef your_frozen_model.pb --output output_model.onnx --inputs input0:0,input1:0... --outputs output0:0,output1:0,output2:0...
    ```
    args follow inputs and outputs are the names of input and output nodes in .pb graph repectly, for example, if the input node name is **x** and output node name is **y1**,**y2**, then the convert bash should be:
    ```
    python -m tf2onnx.convert --graphdef your_frozen_model.pb --output output_model.onnx --inputs x:0 --outputs y1:0,y2:0
    ```

**congratulation! now you are able to use the onnx model for Hyperpose prediction library.**

