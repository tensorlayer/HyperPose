# Quick Start of Training Library

## Prerequisites
* Make sure you have configured 'hyperpose' virtual environment following the training installation guide,(if not, you can refer to [training installation](../install/training.md)).
* Make sure your GPU is available now(using tf.test.is_gpu_available() and it should return True)
* Make sure the hyperpose training Library is under the root directory of the project(where you write train.py and eval.py) or you have installed hyperpose through pypi.

## Train a model
The training procedure of Hyperpose is to set the model architecture, model backbone and dataset.
User specify these configuration using the set up functions of *Config* module with predefined enum value.
The code for training as simple as following would work.
```bash
# >>> import modules of hyperpose
from hyperpose import Config,Model,Dataset
# >>> set model name is necessary to distinguish models (neccesarry)
Config.set_model_name("my_lopps")
# >>> set model architecture (and set model backbone when in need)
Config.set_model_type(Config.MODEL.LightweightOpenpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# >>> set dataset to use
Config.set_dataset_type(Config.DATA.MSCOCO)
# >>> set training type 
Config.set_train_type(Config.TRAIN.Single_train)
# >>> configuration is done, get config object and assemble the system
config=Config.get_config()
model=Model.get_model(config)
dataset=Dataset.get_dataset(config)
train=Model.get_train(config)
# >>> train!
train(model,dataset)
```
Then the integrated training pipeline will start.
for each model, Hyperpose will save all the related files in the directory:
*./save_dir/model_name*, where *model_name* is the name user set by using *Config.set_model_name*
the directory and its contents are below:  
* directory to save model                      ./save_dir/model_name/model_dir  
* directory to save train result               ./save_dir/model_name/train_vis_dir  
* directory to save evaluate result            ./save_dir/model_name/eval_vis_dir  
* directory to save test result                ./save_dir/model_name/test_vis_dir  
* directory to save dataset visualize result   ./save_dir/model_name/data_vis_dir  
* file path to save train log                  ./save_dir/model_name/log.txt  

We provide a helpful training script with cli located at [train.py](https://github.com/tensorlayer/hyperpose/blob/master/train.py) to demonstrate the usage of hyperpose python training library, users can directly use the script to train thier own model or use it as a template for further modification.

## Eval a model
The evaluate procedure using Hyperpose is almost the same to the training procedure,
the model will be loaded from the ./save_dir/model_name/model_dir/newest_model.npz,
The code for evaluating is followed:
```bash
# >>> import modules of hyperpose
from hyperpose import Config,Model,Dataset
# >>> set model name to be eval
Config.set_model_name(args.model_name)
# >>> the model architecture and backbone setting should be the same with the training configuration of the model to be evaluated.
Config.set_model_type(Config.MODEL.LightweightOpenpose)
Config.set_model_backbone(Config.BACKBONE.Vggtiny)
# >>> set dataset to use
Config.set_dataset_type(Config.DATA.MSCOCO)
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

We also provide a helpful evaluating script with cli located at [eval.py](https://github.com/tensorlayer/hyperpose/blob/master/eval.py) to demonstrate how to evaluate the model trained by hyperpose, users can directly use the script to evaluate thier own model or use it as a template for further modification.

The above code sections show the simplest way to use Hyperpose training library to train and evaluate a model trained by Hyperpose, to make full use of Hyperpose training library, you can refer to [training tutorial](../tutorial/training.md)

## Export a model
The trained model weight is saved as a .npz file. For further deployment, one should convert the model loaded with the well-trained weight saved in the .npz file and convert it into the .pb format and .onnx format.
To export a model trained by Hyperpose, one should follow two step:
* (1)convert the trained .npz model into .pb format
    We use the *@tf.function* decorator to produce the static computation graph and save it into the .pb format.
    We already provide a script with cli to facilitate conversion, which located at [export_pb.py](https://github.com/tensorlayer/hyperpose/blob/master/export_pb.py). 
    To convert a model with model_type=**your_model_type** and model_name=**your_model_name** developed by hyperpose,one should place the trained model weight **newest_model.npz** file at path *./save_dir/your_model_name/model_dir/newest_model.npz*,and run the command line followed:
    ```bash
        python export_pb.py --model_type=your_model_type --model_name=your_model_name
    ```
    Then the **frozen_your_model_name.pb** will be produced at path *./save_dir/your_model_name/frozen_your_model_name.pb*.
    one can also export by loading model and using *get_concrete_function* by himself, please refer the [tutorial](../tutorial/training.md) for more details.
* (2)convert the frozen .pb format model into .onnx format
    We use *tf2onnx* library to convert the .pb format model into .onnx format.
    Make sure you have installed the extra requirements for exporting models from [training installation](../install/training.md).<br>
    if you don't know the input and output node names of the pb model,you should use the function *summarize_graph* function 
    of *graph_transforms* from tensorflow. (see [tensorflow tools](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/graph_transforms#using-the-graph-transform-tool) for more details.)

    ```bash
    bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=your_frozen_model.pb
    ```
    then, after knowing the input and output nodes of your .pb model,use tf2onnx
    ```bash
    python -m tf2onnx.convert --graphdef your_frozen_model.pb --output output_model.onnx --inputs input0:0,input1:0... --outputs output0:0,output1:0,output2:0...
    ```
    args follow *--inputs* and *-outputs* are the names of input and output nodes in .pb graph respectively, for example, if the input node name is **x** and output node name is **y1**,**y2**, then the convert bash command line should be:
    ```
    python -m tf2onnx.convert --graphdef your_frozen_model.pb --output output_model.onnx --inputs x:0 --outputs y1:0,y2:0
    ```

**congratulation! now you are able to use the onnx model for Hyperpose prediction library.**

