import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
from .metrics import MetricManager

class BasicModel(Model):
    def __init__(self,config, *args, **kargs):
        super().__init__()
        self.config=config
    
    @tf.function(experimental_relax_shapes=True)
    def forward(self, x, is_train=True, ret_backbone=False):
        '''custom model forwarding
        The `forward` function is expected to take in the images and return the activation maps calculated by the neural network. Implement your custom forwarding computation logic here.
        
        Parameters
        ----------
        x : tf.Tensor
            The input batch of images
        is_train : bool
            a bool value indicate that whether the forward function is invoked at training time or inference time. In the training procedure, the forward function will be called with `is_train=True`; while in the inference procedure and model exportation procedure, the forward function will be called with `is_train=False`. This is because in training, it's better to arrange the model output results in dict by their names, while in inference and model exportation procedure, the model forwarding output results are required to be tensors, especially in model exportation.
        ret_backbone : bool
            a bool value indicate that whether the forward function will output the backbone extracted feature maps when invokded. If `ret_backbone` is set to be False, the model output should be the original final forwarding result(noted as `output`), else if `ret_backbone` is set to be False, the model output should include the backbone extracted feature maps besides the final forwarding result(noted as `output,backbone_features`). This is use for domain adaptation, which need to use the discriminator to align the feature maps extracted by the model backbone facing the labeled training data and unlabeled training data, thus extend the model ability towards the extra unlabled datasets.

        '''
        raise NotImplementedError("virtual class BaseModel function: `forward` not implemented!")
    
    @tf.function(experimental_relax_shapes=True)
    def infer(self, x):
        '''custom model inference
        The `infer`  function is expected to take in the images and return the activation maps calculated by the neural network. Implement your custom inference computation logic here. The difference between the `forward` function and the `infer` function is that, the `infer` function is invoked especially in model exportation procedure, thus it's outputs are required to be the decoded `tf.Tensor` variables. In this way, The `infer` function is usually a warp function of the `forward` function, and the `infer` function output is usually constracted by parsing and formatting the `forward` function result into the `tf.Tensor` format. 
        '''
        raise NotImplementedError("virtual class BaseModel function: `infer` not implemented!")
    
    def cal_loss(self, predict_x, target_x, metric_manager:MetricManager ,mask=None):
        '''custom loss calculation function
        Teh `cal_loss` function is expected to take the output predict activation map and the ground truth target activation map and return the calculated loss for gradient descent.

        Parameters:
        ----------
        predict_x : Dictionary
            a dictionary contains the model predict activation map, the keys are the activation map name and the values are the corresponding activation map value. `predict_x` should be the output of the `forward` function.
        target_x : Dicyionary
            a dictionary contains the ground truth activation map, the keys are the activation map name and the values are the corresponding activation map value.  `target_x` should have the same keys as the `predict_x` but the corresponding value of `target_x` should be the ground truth while corresponding value of the `predict_x` should be the model predict value.
        '''
        raise NotImplementedError("virtual class BaseModel function: `cal_loss` not implemented!")