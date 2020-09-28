import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from ..backbones import Resnet50_backbone

class Pifpaf(Model):
    def __init__(self,parts,limbs,n_pos=18,n_limbs=19,hin=368,win=368,backbone=None,pretraining=False,quad_size=2,data_format="channels_first"):
        super().__init__()
        self.parts=parts
        self.limbs=limbs
        self.n_pos=n_pos
        self.n_limbs=n_limbs
        self.quad_size=quad_size
        self.data_format=data_format
        if(backbone==None):
            self.backbone=Resnet50_backbone(data_format=data_format,use_pool=False)
        else:
            self.backbone=backbone(data_format=data_format)
        self.pif_head=self.PifHead(input_features=self.backbone.out_channels,n_pos=self.n_pos,n_limbs=self.n_limbs,\
            quad_size=self.quad_size,data_format=self.data_format)
        self.paf_head=self.PafHead(input_features=self.backbone.out_channels,n_pos=self.n_pos,n_limbs=self.n_limbs,\
            quad_size=self.quad_size,data_format=self.data_format)
    
    def forward(self,x):
        x=self.backbone.forward(x)
        pif_map=self.pif_head.forward(x)
        paf_map=self.paf_head.forward(x)
        return pif_map,paf_map
    
    class PifHead(Model):
        def __init__(self,input_features=2048,n_pos=19,n_limbs=19,quad_size=2,data_format="channels_first"):
            self.input_features=input_features
            self.n_pos=n_pos
            self.n_limbs=n_limbs
            self.quad_size=quad_size
            self.out_features=self.n_pos*5*(2**self.quad_size)
            self.data_format=data_format
            self.tf_data_format="NCHW" if self.data_format=="channels_first" else "NHWC"
            self.main_block=Conv2d(n_filter=self.out_features,in_channels=self.input_features,filter_size=(1,1),data_format=self.data_format)

        def forward(self,x):
            x=self.main_block.forward(x)
            x=tf.nn.depth_to_space(x,block_size=self.quad_size,data_format=self.tf_data_format)
            x=tf.reshape(x,[x.shape[0],self.n_pos,5,x.shape[2],x.shape[3]])
            pif_conf=x[:,:,0:1,:,:]
            pif_vec=x[:,:,1:3,:,:]
            pif_logb=x[:,:,3:4,:,:]
            pif_scale=x[:,:,4:5,:,:]
            return pif_conf,pif_vec,pif_logb,pif_scale
        
    class PafHead(Model):
        def __init__(self,input_features=2048,n_pos=19,n_limbs=19,quad_size=2,data_format="channels_first"):
            self.input_features=input_features
            self.n_pos=n_pos
            self.n_limbs=n_limbs
            self.quad_size=quad_size
            self.out_features=self.n_limbs*9*(2**self.quad_size)
            self.data_format=data_format
            self.tf_data_format="NCHW" if self.data_format=="channels_first" else "NHWC"
            self.main_block=Conv2d(n_filter=self.out_features,in_channels=self.input_features,filter_size=(1,1),data_format=self.data_format)
        
        def forward(self,x):
            x=self.main_block.forward(x)
            x=tf.nn.depth_to_space(x,block_size=self.quad_size,data_format=self.tf_data_format)
            x=tf.reshape(x,[x.shape[0],self.n_limbs,9,x.shape[2],x.shape[3]])
            paf_conf=x[:,:,0:1,:,:]
            paf_vec1=x[:,:,1:3,:,:]
            paf_vec2=x[:,:,3:5,:,:]
            paf_logb1=x[:,:,5:6,:,:]
            paf_logb2=x[:,:,6:7,:,:]
            paf_scale1=x[:,:,7:8,:,:]
            paf_scale2=x[:,:,8:9,:,:]
            return paf_conf,paf_vec1,paf_vec2,paf_logb1,paf_logb2,paf_scale1,paf_scale2
