import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d, SeparableConv2d,\
        MeanPool2d, Dense, Flatten, UpSampling2d

class MobilenetV1_backbone(Model):
    def __init__(self,scale_size=8,data_format="channels_last",pretraining=False):
        super().__init__()
        self.name="MobilenetV1_backbone"
        self.data_format=data_format
        self.scale_size=scale_size
        self.pretraining=pretraining
        self.main_layer_list=[]
        if(self.scale_size==8 or self.scale_size==32 or self.pretraining):
            self.main_layer_list+=self.conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(2,2),name="block_1")
            self.main_layer_list+=self.separable_conv_block(n_filter=64,in_channels=32,filter_size=(3,3),strides=(1,1),name="block_2")
            self.main_layer_list+=self.separable_conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(2,2),name="block_3")
            self.main_layer_list+=self.separable_conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),name="block_4")
            self.main_layer_list+=self.separable_conv_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(2,2),name="block_5")
            self.main_layer_list+=self.separable_conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),name="block_6")
            self.main_layer_list+=self.separable_conv_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=(1,1),name="block_7")
            self.main_layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),name="block_8")
            self.main_layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),name="block_9")
            self.main_layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),name="block_10")
            self.out_channels=512
        if(self.scale_size==32 or self.pretraining):
            self.main_layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(2,2),name="block_11")
            self.main_layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),name="block_12")
            self.main_layer_list+=self.separable_conv_block(n_filter=1024,in_channels=512,filter_size=(3,3),strides=(2,2),name="block_13")
            self.main_layer_list+=self.separable_conv_block(n_filter=1024,in_channels=1024,filter_size=(3,3),strides=(1,1),name="block_14")
            self.out_channels=1024
        if(self.pretraining):
            self.main_layer_list+=[MeanPool2d(filter_size=(7,7),strides=(1,1),data_format=self.data_format,name="meanpool_1")]
            self.main_layer_list+=[Flatten(name="Flatten")]
            self.main_layer_list+=[Dense(n_units=1000,in_channels=1024,act=None,name="dense_1")]
        self.main_block=LayerList(self.main_layer_list)

    def forward(self,x):
        return self.main_block.forward(x)
    
    def cal_loss(self,label,predict):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=predict))

    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),padding="SAME",name="conv_block"):
        layer_list=[]
        layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            data_format=self.data_format,padding=padding,name=f"{name}_conv1"))
        layer_list.append(BatchNorm2d(num_features=n_filter,is_train=True,act=tf.nn.relu,data_format=self.data_format,name=f"{name}_bn1"))
        return layer_list
    
    def separable_conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),name="spconv_block"):
        layer_list=[]
        layer_list.append(DepthwiseConv2d(in_channels=in_channels,filter_size=filter_size,strides=strides,\
            data_format=self.data_format,name=f"{name}_dw1"))
        layer_list.append(BatchNorm2d(num_features=in_channels,is_train=True,act=tf.nn.relu,data_format=self.data_format,name=f"{name}_bn1"))
        layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(1,1),strides=(1,1),data_format=self.data_format,name=f"{name}_dw2"))
        layer_list.append(BatchNorm2d(num_features=n_filter,is_train=True,act=tf.nn.relu,data_format=self.data_format,name=f"{name}_bn2"))
        return layer_list

class MobilenetV2_backbone(Model):
    def __init__(self,scale_size=8,data_format="channels_last",pretraining=False):
        super().__init__()
        self.name="MobilenetV2_backbone"
        self.data_format=data_format
        self.scale_size=scale_size
        self.pretraining=pretraining
        self.main_layer_list=[]
        if(self.scale_size==8 or self.scale_size==32 or self.pretraining):
            #block_1 n=1
            self.block_1_1=Conv2d(n_filter=32,in_channels=3,filter_size=(3,3),strides=(2,2),data_format=self.data_format,name="block1_conv1")
            self.block_1_2=BatchNorm2d(num_features=32,is_train=True,act=tf.nn.relu6,data_format=self.data_format,name="blcok1_bn1")
            #block_2 n=1
            self.block_2_1=self.InvertedResidual(n_filter=16,in_channels=32,strides=(1,1),exp_ratio=1,data_format=self.data_format,name="block2")
            #block_3 n=2
            self.block_3_1=self.InvertedResidual(n_filter=24,in_channels=16,strides=(2,2),exp_ratio=6,data_format=self.data_format,name="block3_1")
            self.block_3_2=self.InvertedResidual(n_filter=24,in_channels=24,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block3_2")
            #block_4 n=3
            self.block_4_1=self.InvertedResidual(n_filter=32,in_channels=24,strides=(2,2),exp_ratio=6,data_format=self.data_format,name="block4_1")
            self.block_4_2=self.InvertedResidual(n_filter=32,in_channels=32,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block4_2")
            self.block_4_3=self.InvertedResidual(n_filter=32,in_channels=32,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block4_3")
            #block_5 n=4
            self.block_5_1=self.InvertedResidual(n_filter=64,in_channels=32,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block5_1")
            self.block_5_2=self.InvertedResidual(n_filter=64,in_channels=64,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block5_2")
            self.block_5_3=self.InvertedResidual(n_filter=64,in_channels=64,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block5_3")
            self.block_5_4=self.InvertedResidual(n_filter=64,in_channels=64,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block5_4")
            self.out_channels=64
        if(self.scale_size==32 or self.pretraining):
            #block_6 n=3
            self.block_6_1=self.InvertedResidual(n_filter=96,in_channels=64,strides=(2,2),exp_ratio=6,data_format=self.data_format,name="block6_1")
            self.block_6_2=self.InvertedResidual(n_filter=96,in_channels=96,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block6_2")
            self.block_6_3=self.InvertedResidual(n_filter=96,in_channels=96,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block6_3")
            #block_7 n=3
            self.block_7_1=self.InvertedResidual(n_filter=160,in_channels=96,strides=(2,2),exp_ratio=6,data_format=self.data_format,name="block7_1")
            self.block_7_2=self.InvertedResidual(n_filter=160,in_channels=160,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block7_2")
            self.block_7_3=self.InvertedResidual(n_filter=160,in_channels=160,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block7_3")
            #block_8 n=1
            self.block_8=self.InvertedResidual(n_filter=320,in_channels=160,strides=(1,1),exp_ratio=6,data_format=self.data_format,name="block8")
            self.out_channels=320
        if(self.pretraining):
            self.block_9_1=Conv2d(n_filter=1280,in_channels=320,filter_size=(1,1),strides=(1,1),data_format=self.data_format,name="block9_conv1")
            self.block_9_2=MeanPool2d(filter_size=(7,7),strides=(1,1),data_format=self.data_format,name="block9_pool1")
            self.block_9_3=Conv2d(n_filter=1000,in_channels=1280,filter_size=(1,1),strides=(1,1),act=None,data_format=self.data_format,name="block9_conv2")
            self.block_9_4=Flatten(name="Flatten")

    def forward(self,x):
        if(self.scale_size==8 or self.scale_size==32 or self.pretraining):
            x=self.block_1_1.forward(x)
            x=self.block_1_2.forward(x)
            x=self.block_2_1.forward(x)
            x=self.block_3_1.forward(x)
            x=self.block_3_2.forward(x)
            x=self.block_4_1.forward(x)
            x=self.block_4_2.forward(x)
            x=self.block_4_3.forward(x)
            x=self.block_5_1.forward(x)
            x=self.block_5_2.forward(x)
            x=self.block_5_3.forward(x)
            x=self.block_5_4.forward(x)
        if(self.scale_size==32 or self.pretraining):
            x=self.block_6_1.forward(x)
            x=self.block_6_2.forward(x)
            x=self.block_6_3.forward(x)
            x=self.block_7_1.forward(x)
            x=self.block_7_2.forward(x)
            x=self.block_7_3.forward(x)
            x=self.block_8.forward(x)
        if(self.pretraining):
            x=self.block_9_1.forward(x)
            x=self.block_9_2.forward(x)
            x=self.block_9_3.forward(x)
        return x

    def cal_loss(self,label,predict):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=predict))

    class InvertedResidual(Model):
        def __init__(self,n_filter=128,in_channels=128,strides=(1,1),exp_ratio=6,data_format="channels_first",name="block"):
            super().__init__()
            self.n_filter=n_filter
            self.in_channels=in_channels
            self.strides=strides
            self.exp_ratio=exp_ratio
            self.data_format=data_format
            self.name=name
            self.hidden_dim=self.exp_ratio*self.in_channels
            self.identity=False
            if(self.strides==(1,1) and self.in_channels==self.n_filter):
                self.identity=True
            if(self.exp_ratio==1):
                self.main_block=LayerList([
                    DepthwiseConv2d(in_channels=self.hidden_dim,filter_size=(3,3),strides=self.strides,\
                        b_init=None,data_format=self.data_format,name=f"{self.name}_conv1"),
                    BatchNorm2d(num_features=self.hidden_dim,is_train=True,act=tf.nn.relu6,data_format=self.data_format,name=f"{self.name}_bn1"),
                    Conv2d(n_filter=self.n_filter,in_channels=self.hidden_dim,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format,name=f"{self.name}_dw1"),
                    BatchNorm2d(num_features=self.n_filter,is_train=True,act=None,data_format=self.data_format,name=f"{self.name}_bn2")
                ])
            else:
                self.main_block=LayerList([
                    Conv2d(n_filter=self.hidden_dim,in_channels=self.in_channels,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format,name=f"{self.name}_conv1"),
                    BatchNorm2d(num_features=self.hidden_dim,is_train=True,act=tf.nn.relu6,data_format=self.data_format,name=f"{self.name}_bn1"),
                    DepthwiseConv2d(in_channels=self.hidden_dim,filter_size=(3,3),strides=self.strides,\
                        b_init=None,data_format=self.data_format,name=f"{self.name}_dw1"),
                    BatchNorm2d(num_features=self.hidden_dim,is_train=True,act=tf.nn.relu6,data_format=self.data_format,name=f"{self.name}_bn2"),
                    Conv2d(n_filter=self.n_filter,in_channels=self.hidden_dim,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format,name=f"{self.name}_conv2")
                ])

        def forward(self,x):
            if(self.identity):
                return x+self.main_block.forward(x)
            else:
                return self.main_block.forward(x)

initializer=tl.initializers.truncated_normal(stddev=0.005)
def conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),dilation_rate=(1,1),W_init=initializer,b_init=initializer,padding="SAME",data_format="channels_first"):
    layer_list=[]
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=filter_size,strides=strides,in_channels=in_channels,\
        dilation_rate=dilation_rate,padding=padding,W_init=initializer,b_init=initializer,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99, act=tf.nn.relu,num_features=n_filter,data_format=data_format,is_train=True))
    return layers.LayerList(layer_list)
    
def dw_conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),dilation_rate=(1,1),W_init=initializer,b_init=initializer,data_format="channels_first"):
    layer_list=[]
    layer_list.append(DepthwiseConv2d(filter_size=filter_size,strides=strides,in_channels=in_channels,
        dilation_rate=dilation_rate,W_init=initializer,b_init=None,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99,act=tf.nn.relu,num_features=in_channels,data_format=data_format,is_train=True))
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=(1,1),strides=(1,1),in_channels=in_channels,W_init=initializer,b_init=None,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99,act=tf.nn.relu,num_features=n_filter,data_format=data_format,is_train=True))
    return layers.LayerList(layer_list)

def nobn_dw_conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),W_init=initializer,b_init=initializer,data_format="channels_first"):
    layer_list=[]
    layer_list.append(DepthwiseConv2d(filter_size=filter_size,strides=strides,in_channels=in_channels,
        act=tf.nn.relu,W_init=initializer,b_init=None,data_format=data_format))
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=(1, 1),strides=(1, 1),in_channels=in_channels,
        act=tf.nn.relu,W_init=initializer,b_init=None,data_format=data_format))
    return layers.LayerList(layer_list)
    
class MobilenetDilated_backbone(Model):
    def __init__(self,scale_size=8, data_format="channels_first", pretraining=False):
        super().__init__()
        self.scale_size = scale_size
        self.data_format = data_format
        self.pretraining = pretraining
        self.out_channels=512
        self.scale_size=8
        if(self.scale_size==8):
            strides=(1,1)
        elif(self.scale_size==32 or self.pretraining):
            strides=(2,2)
        self.main_block=layers.LayerList([
        conv_block(n_filter=32,in_channels=3,data_format=self.data_format,strides=(2,2)),
        dw_conv_block(n_filter=64,in_channels=32,data_format=self.data_format),
        dw_conv_block(n_filter=128,in_channels=64,data_format=self.data_format,strides=(2,2)),
        dw_conv_block(n_filter=128,in_channels=128,data_format=self.data_format),
        dw_conv_block(n_filter=256,in_channels=128,data_format=self.data_format,strides=(2,2)),
        dw_conv_block(n_filter=256,in_channels=256,data_format=self.data_format),
        dw_conv_block(n_filter=512,in_channels=256,data_format=self.data_format),
        dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format,dilation_rate=(2,2), strides=strides),
        dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format),
        dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format, strides=strides),
        dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format),
        dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format)
        ])

    def forward(self,x):
        return self.main_block.forward(x)

initial_w=tl.initializers.random_normal(stddev=0.01)
initial_b=tl.initializers.constant(value=0.0)

def conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,padding="SAME",data_format="channels_first"):
    layer_list=[]
    layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,act=act,\
        W_init=initial_w,b_init=initial_b,data_format=data_format,padding=padding))
    layer_list.append(BatchNorm2d(num_features=n_filter,decay=0.999,is_train=True,act=act,data_format=data_format))
    return LayerList(layer_list)

def separable_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),dilation_rate=(1,1),act=tf.nn.relu,data_format="channels_first"):
    layer_list=[]
    layer_list.append(DepthwiseConv2d(filter_size=filter_size,strides=strides,in_channels=in_channels,
        dilation_rate=dilation_rate,W_init=initial_w,b_init=None,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99,act=act,num_features=in_channels,data_format=data_format,is_train=True))
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=(1,1),strides=(1,1),in_channels=in_channels,W_init=initial_w,b_init=None,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99,act=act,num_features=n_filter,data_format=data_format,is_train=True))
    return layers.LayerList(layer_list)

class MobilenetThin_backbone(Model):
    def __init__(self,scale_size=8,data_format="channels_first", pretraining=False):
        super().__init__()
        self.scale_size = scale_size
        self.data_format = data_format
        self.pretraining = pretraining
        self.out_channels=1152
        if(self.data_format=="channels_first"):
            self.concat_dim=1
        else:
            self.concat_dim=-1
        if(self.scale_size==8):
            strides=(1,1)
        elif(self.scale_size==32 or self.pretraining):
            strides=(2,2)
        self.convblock_0=conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_1=separable_block(n_filter=64,in_channels=32,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_2=separable_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_3=separable_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_4=separable_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_5=separable_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_6=separable_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=strides,act=tf.nn.relu,data_format=self.data_format)
        self.convblock_7=separable_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_8=separable_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_9=separable_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=strides,act=tf.nn.relu,data_format=self.data_format)
        self.convblock_10=separable_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_11=separable_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.maxpool=MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format)
    
    def forward(self,x):
        concat_list=[]
        x=self.convblock_0.forward(x)
        x=self.convblock_1.forward(x)
        x=self.convblock_2.forward(x)
        x=self.convblock_3.forward(x)
        concat_list.append(self.maxpool.forward(x))
        x=self.convblock_4.forward(x)
        x=self.convblock_5.forward(x)
        x=self.convblock_6.forward(x)
        x=self.convblock_7.forward(x)
        concat_list.append(x)
        x=self.convblock_8.forward(x)
        x=self.convblock_9.forward(x)
        x=self.convblock_10.forward(x)
        x=self.convblock_11.forward(x)
        concat_list.append(x)
        x=tf.concat(concat_list,self.concat_dim)
        return x

class MobilenetSmall_backbone:
    def __init__(self,scale_size=8,data_format="channels_first", pretraining=False):
        super().__init__()
        self.scale_size = scale_size
        self.data_format = data_format
        self.pretraining = pretraining
        if(self.data_format=="channels_first"):
            self.concat_dim=1
        else:
            self.concat_dim=-1
        self.out_channels=704
        
        if(self.scale_size == 8):
            strides=(1,1)
        elif(self.scale_size == 32 or self.pretraining):
            strides=(2,2)

        self.convblock_0=conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_1=separable_block(n_filter=64,in_channels=32,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_2=separable_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_3=separable_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_4=separable_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_5=separable_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
        self.convblock_6=separable_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=strides,act=tf.nn.relu,data_format=self.data_format)
        self.convblock_7=separable_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=strides,act=tf.nn.relu,data_format=self.data_format)
        self.maxpool=MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format)
        self.upsample=UpSampling2d(scale=2,data_format=self.data_format)

    def forward(self,x):
        concat_list=[]
        x=self.convblock_0.forward(x)
        x=self.convblock_1.forward(x)
        concat_list.append(self.maxpool.forward(x))
        x=self.convblock_2.forward(x)
        x=self.convblock_3.forward(x)
        concat_list.append(x)
        x=self.convblock_4.forward(x)
        x=self.convblock_5.forward(x)
        x=self.convblock_6.forward(x)
        x=self.convblock_7.forward(x)
        concat_list.append(self.upsample.forward(x))
        x=tf.concat(concat_list,self.concat_dim)
        return x

class vggtiny_backbone(Model):
    def __init__(self,in_channels=3,scale_size=8,data_format="channels_first",pretraining=False):
        super().__init__()
        self.name="vggtiny_backbone"
        self.in_channels=in_channels
        self.data_format=data_format
        self.scale_size=scale_size
        self.pretraining=pretraining
        self.main_layer_list=[]
        if(self.scale_size==8 or self.scale_size==32 or self.pretraining):
            self.main_layer_list+=self.conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),name="block_1_1")
            self.main_layer_list+=self.conv_block(n_filter=64,in_channels=32,filter_size=(3,3),strides=(1,1),name="block_1_2")
            self.main_layer_list+=[MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format,name="maxpool_1")]
            self.main_layer_list+=self.conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(1,1),name="block_2_1")
            self.main_layer_list+=self.conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),name="block_2_2")
            self.main_layer_list+=[MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format,name="maxpool_2")]
            self.main_layer_list+=self.conv_block(n_filter=200,in_channels=128,filter_size=(3,3),strides=(1,1),name="block_3_1")
            self.main_layer_list+=self.conv_block(n_filter=200,in_channels=200,filter_size=(3,3),strides=(1,1),name="block_3_2")
            self.main_layer_list+=self.conv_block(n_filter=200,in_channels=200,filter_size=(3,3),strides=(1,1),name="block_3_3")
            self.main_layer_list+=[MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format,name="maxpool_3")]
            self.main_layer_list+=self.conv_block(n_filter=384,in_channels=200,filter_size=(3,3),strides=(1,1),name="block_4_1")
            self.main_layer_list+=self.conv_block(n_filter=384,in_channels=384,filter_size=(3,3),strides=(1,1),name="block_4_2")
            self.out_channels=384
        if(self.scale_size==32 or self.pretraining):
            self.main_layer_list+=self.conv_block(n_filter=384,in_channels=384,filter_size=(3,3),strides=(2,2),name="block_4_3")
            self.main_layer_list+=self.conv_block(n_filter=384,in_channels=384,filter_size=(3,3),strides=(1,1),name="block_4_4")
            self.main_layer_list+=self.conv_block(n_filter=384,in_channels=384,filter_size=(3,3),strides=(2,2),name="block_4_5")
            self.out_channels=384
        if(self.pretraining):
            self.main_layer_list+=[
                Flatten(name="Flatten"),
                Dense(n_units=4096,in_channels=384*7*7,act=tf.nn.relu,name="fc1"),
                Dense(n_units=4096,in_channels=4096,act=tf.nn.relu,name="fc2"),
                Dense(n_units=1000,in_channels=4096,act=None,name="fc3")
            ]
        self.main_block=LayerList(self.main_layer_list)

    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,padding="SAME",name="block"):
        layer_list=[]
        layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            act=None,data_format=self.data_format,padding=padding,name=f"{name}_conv1"))
        layer_list.append(BatchNorm2d(num_features=n_filter,act=act,is_train=True,data_format=self.data_format,name=f"{name}_bn1"))
        return layer_list
    
    def forward(self,x):
        return self.main_block.forward(x)
    
    def cal_loss(self,label,predict):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=predict))

class vgg16_backbone(Model):
    def __init__(self,in_channels=3,scale_size=8,data_format="channels_first",pretraining=False):
        super().__init__()
        self.name="vgg16_backbone"
        self.in_channels=in_channels
        self.data_format=data_format
        self.scale_size=scale_size
        self.pretraining=pretraining
        self.main_layer_list=[]
        if(self.scale_size==8 or self.scale_size==32 or self.pretraining):
            self.main_layer_list+=[
                self.conv_block(n_filter=64,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_1_1"),
                self.conv_block(n_filter=64,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_1_2"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_1"),
                self.conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_2_1"),
                self.conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_2_2"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_2"),
                self.conv_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_3_1"),
                self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_3_2"),
                self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_3_3"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_3"),
                self.conv_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_4_1"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_4_2"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_4_3")
            ]
            self.out_channels=512
        if(self.scale_size==32 or self.pretraining):
            self.main_layer_list+=[
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_4"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_5_1"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_5_2"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="block_5_3"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_5")
            ]
            self.out_channels=512
        if(self.pretraining):
            self.main_layer_list+=[
                Flatten(name="Flatten"),
                Dense(n_units=4096,in_channels=512*7*7,act=tf.nn.relu,name="fc1"),
                Dense(n_units=4096,in_channels=4096,act=tf.nn.relu,name="fc2"),
                Dense(n_units=1000,in_channels=4096,act=None,name="fc3")
            ]
        self.main_block=LayerList(self.main_layer_list)

    def forward(self,x):
        return self.main_block.forward(x)
    
    def cal_loss(self,label,predict):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=predict))
    
    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=None,padding="SAME",name="block"):
        return Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            act=act,data_format=self.data_format,padding=padding,name=f"{name}_conv1")

class vgg19_backbone(Model):
    def __init__(self,in_channels=3,scale_size=8,data_format="channels_first",pretraining=False):
        super().__init__()
        self.name="vgg19_backbone"
        self.in_channels=in_channels
        self.data_format=data_format
        self.scale_size=scale_size
        self.pretraining=pretraining
        self.vgg_mean=tf.constant([103.939, 116.779, 123.68])/255
        if(self.data_format=="channels_first"):
            self.vgg_mean=tf.reshape(self.vgg_mean,[1,3,1,1])
        elif(self.data_format=="channels_last"):
            self.vgg_mean=tf.reshape(self.vgg_mean,[1,1,1,3])
        self.initializer=tl.initializers.truncated_normal(stddev=0.005)
        self.main_layer_list=[]
        if(self.scale_size==8 or self.scale_size==32 or self.pretraining):
            self.main_layer_list+=[
                self.conv_block(n_filter=64,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv1_1"),
                self.conv_block(n_filter=64,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv1_2"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_1"),
                self.conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv2_1"),
                self.conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv2_2"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_2"),
                self.conv_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv3_1"),
                self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv3_2"),
                self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv3_3"),
                self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv3_4"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_3"),
                self.conv_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv4_1"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv4_2")
            ]
            self.out_channels=512
        if(self.scale_size==32 or self.pretraining):
            self.main_layer_list+=[
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv4_3"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv4_4"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_4"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv5_1"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv5_2"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv5_3"),
                self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,name="conv5_4"),
                MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format,name="maxpool_5")
            ]
            self.out_channels=512
        if(self.pretraining):
            self.main_layer_list+=[
                Flatten(name="Flatten"),
                Dense(n_units=4096,in_channels=512*7*7,act=tf.nn.relu,W_init=self.initializer,b_init=self.initializer,name="fc6"),
                Dense(n_units=4096,in_channels=4096,act=tf.nn.relu,W_init=self.initializer,b_init=self.initializer,name="fc7"),
                Dense(n_units=1000,in_channels=4096,act=None,W_init=self.initializer,b_init=self.initializer,name="fc8")
            ]
        self.main_block=LayerList(self.main_layer_list)

    def forward(self,x):
        x=x-self.vgg_mean
        return self.main_block.forward(x)
    
    def cal_loss(self,label,predict):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=predict))
    
    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=None,padding="SAME",name="conv_default"):
        return Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            act=act,data_format=self.data_format,padding=padding,W_init=self.initializer,b_init=self.initializer,name=name)


class Resnet18_backbone(Model):
    def __init__(self,n_filter=512,in_channels=3,scale_size=8,data_format="channels_first",pretraining=False):
        super().__init__()
        self.name="resnet18_backbone"
        self.data_format=data_format
        self.out_channels=n_filter
        self.scale_size=scale_size
        self.pretraining=pretraining
        self.out_channels=512
        if(self.scale_size==8):
            strides=(1,1)
        elif(self.scale_size==32 or self.pretraining):
            strides=(2,2)
        self.conv1=Conv2d(n_filter=64,in_channels=in_channels,filter_size=(7,7),strides=(2,2),b_init=None,data_format=self.data_format,name="conv_1_1")
        self.bn1=BatchNorm2d(decay=0.9,act=tf.nn.relu,is_train=True,num_features=64,data_format=self.data_format,name="bn_1_1")
        self.maxpool=MaxPool2d(filter_size=(3,3),strides=(2,2),data_format=self.data_format,name="maxpool_1")
        self.res_block_2_1=self.Res_block(n_filter=64,in_channels=64,strides=(1,1),is_down_sample=False,data_format=self.data_format,name="block_2_1")
        self.res_block_2_2=self.Res_block(n_filter=64,in_channels=64,strides=(1,1),is_down_sample=False,data_format=self.data_format,name="block_2_2")
        self.res_block_3_1=self.Res_block(n_filter=128,in_channels=64,strides=(2,2),is_down_sample=True,data_format=self.data_format,name="block_3_1")
        self.res_block_3_2=self.Res_block(n_filter=128,in_channels=128,strides=(1,1),is_down_sample=False,data_format=self.data_format,name="block_3_2")
        self.res_block_4_1=self.Res_block(n_filter=256,in_channels=128,strides=strides,is_down_sample=True,data_format=self.data_format,name="block_4_1")
        self.res_block_4_2=self.Res_block(n_filter=256,in_channels=256,strides=(1,1),is_down_sample=False,data_format=self.data_format,name="block_4_2")
        self.res_block_5_1=self.Res_block(n_filter=512,in_channels=256,strides=strides,is_down_sample=True,data_format=self.data_format,name="block_5_1")
        if(self.pretraining):
            self.res_block_5_2=self.Res_block(n_filter=512,in_channels=512,strides=(1,1),is_down_sample=False,data_format=self.data_format,name="block_5_2")
            self.avg_pool=MeanPool2d(filter_size=(7,7),strides=(1,1),data_format=self.data_format,name="avgpool_2")
            self.flatten=Flatten(name="Flatten")
            self.fc=Dense(n_units=1000,in_channels=512,name="FC")

    def forward(self,x):
        x=self.conv1.forward(x)
        x=self.bn1.forward(x)
        x=self.maxpool.forward(x)
        x=self.res_block_2_1.forward(x)
        x=self.res_block_2_2.forward(x)
        x=self.res_block_3_1.forward(x)
        x=self.res_block_3_2.forward(x)
        x=self.res_block_4_1.forward(x)
        x=self.res_block_4_2.forward(x)
        x=self.res_block_5_1.forward(x)
        if(self.pretraining):
            x=self.res_block_5_2.forward(x)
            x=self.avg_pool.forward(x)
            x=self.flatten.forward(x)
            x=self.fc.forward(x)
        return x

    class Res_block(Model):
        def __init__(self,n_filter,in_channels,strides=(1,1),is_down_sample=False,data_format="channels_first",name="res_block"):
            super().__init__()
            self.data_format=data_format
            self.is_down_sample=is_down_sample
            if(is_down_sample):
                init_filter_size=(1,1)
            else:
                init_filter_size=(3,3)
            self.main_block=LayerList([
            Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(3,3),strides=strides,b_init=None,data_format=self.data_format,name=f"{name}_conv_1"),
            BatchNorm2d(decay=0.9,act=tf.nn.relu,is_train=True,num_features=n_filter,data_format=self.data_format,name=f"{name}_bn_1"),
            Conv2d(n_filter=n_filter,in_channels=n_filter,filter_size=(3,3),strides=(1,1),b_init=None,data_format=self.data_format,name=f"{name}_conv_2"),
            BatchNorm2d(decay=0.9,is_train=True,num_features=n_filter,data_format=self.data_format,name=f"{name}_bn_2"),
            ])
            if(self.is_down_sample):
                self.down_sample=LayerList([
                    Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=init_filter_size,strides=strides,b_init=None,data_format=self.data_format,name=f"{name}_downsample_conv"),
                    BatchNorm2d(decay=0.9,is_train=True,num_features=n_filter,data_format=self.data_format,name=f"{name}_downsample_bn")
                ])

        def forward(self,x):
            res=x
            x=self.main_block.forward(x)
            if(self.is_down_sample):
                res=self.down_sample.forward(res)
            return tf.nn.relu(x+res)   

class Resnet50_backbone(Model):
    def __init__(self,in_channels=3,n_filter=64,scale_size=8,decay=0.9,eps=1e-5,data_format="channels_first",pretraining=False,use_pool=True):
        super().__init__()
        self.name="resnet50_backbone"
        self.in_channels=in_channels
        self.n_filter=n_filter
        self.scale_size=scale_size
        self.data_format=data_format
        self.pretraining=pretraining
        self.out_channels=2048
        self.use_pool=use_pool
        if(self.scale_size==8):
            strides=(1,1)
        elif(self.scale_size==32 or self.pretraining):
            strides=(2,2)
        self.eps=eps
        self.decay=decay
        #first layers
        self.conv1=Conv2d(n_filter=64,in_channels=self.in_channels,filter_size=(7,7),strides=(2,2),padding="SAME",b_init=None,data_format=self.data_format,name="conv1")
        self.bn1=BatchNorm2d(decay=self.decay,epsilon=self.eps,is_train=True,num_features=64,data_format=self.data_format,act=tf.nn.relu,name="bn1")
        self.maxpool1=MaxPool2d(filter_size=(3,3),strides=(2,2),data_format=self.data_format,name="maxpool_1")
        #block_1
        self.block_1_1=self.Basic_block(in_channels=64,n_filter=64,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_1_1")
        self.block_1_2=self.Basic_block(in_channels=256,n_filter=64,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_1_2")
        self.block_1_3=self.Basic_block(in_channels=256,n_filter=64,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_1_3")
        #block_2
        self.block_2_1=self.Basic_block(in_channels=256,n_filter=128,strides=(2,2),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_2_1")
        self.block_2_2=self.Basic_block(in_channels=512,n_filter=128,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_2_2")
        self.block_2_3=self.Basic_block(in_channels=512,n_filter=128,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_2_3")
        self.block_2_4=self.Basic_block(in_channels=512,n_filter=128,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_2_4")
        #block_3
        self.block_3_1=self.Basic_block(in_channels=512,n_filter=256,strides=strides,data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_3_1")
        self.block_3_2=self.Basic_block(in_channels=1024,n_filter=256,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_3_2")
        self.block_3_3=self.Basic_block(in_channels=1024,n_filter=256,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_3_3")
        self.block_3_4=self.Basic_block(in_channels=1024,n_filter=256,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_3_4")
        self.block_3_5=self.Basic_block(in_channels=1024,n_filter=256,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_3_5")
        self.block_3_6=self.Basic_block(in_channels=1024,n_filter=256,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_3_6")
        #block_4
        self.block_4_1=self.Basic_block(in_channels=1024,n_filter=512,strides=strides,data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_4_1")
        self.block_4_2=self.Basic_block(in_channels=2048,n_filter=512,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_4_2")
        self.block_4_3=self.Basic_block(in_channels=2048,n_filter=512,strides=(1,1),data_format=self.data_format,eps=self.eps,decay=self.decay,name="block_4_3")
        if(self.pretraining):
            self.block_5=LayerList([
                MeanPool2d(filter_size=(7,7),strides=(1,1),data_format=self.data_format,name="avgpool_1"),
                Flatten(name="Flatten"),
                Dense(n_units=1000,in_channels=2048,act=None,name="fc")
            ])

    def forward(self,x):
        x=self.conv1.forward(x)
        x=self.bn1.forward(x)
        if(self.use_pool):
            x=self.maxpool1.forward(x)
        #block_1
        x=self.block_1_1.forward(x)
        x=self.block_1_2.forward(x)
        x=self.block_1_3.forward(x)
        #block_2 
        x=self.block_2_1.forward(x)
        x=self.block_2_2.forward(x)
        x=self.block_2_3.forward(x)
        x=self.block_2_4.forward(x)
        #block_3
        x=self.block_3_1.forward(x)
        x=self.block_3_2.forward(x)
        x=self.block_3_3.forward(x)
        x=self.block_3_4.forward(x)
        x=self.block_3_5.forward(x)
        x=self.block_3_6.forward(x)
        #block_4
        x=self.block_4_1.forward(x)
        x=self.block_4_2.forward(x)
        x=self.block_4_3.forward(x)
        if(self.pretraining):
            x=self.block_5.forward(x)
        return x

    def cal_loss(self,label,predict):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,logits=predict))

    class Basic_block(Model):
        def __init__(self,in_channels=64,n_filter=64,strides=(1,1),data_format="channels_first",decay=0.9,eps=1e-5,name="basic_block"):
            super().__init__()
            self.in_channels=in_channels
            self.n_filter=n_filter
            self.strides=strides
            self.data_format=data_format
            self.downsample=None
            self.name=name
            self.decay=decay
            self.eps=eps
            if(self.strides!=(1,1) or self.in_channels!=4*self.n_filter):
                self.downsample=LayerList([
                    Conv2d(n_filter=4*self.n_filter,in_channels=self.in_channels,filter_size=(1,1),strides=self.strides,b_init=None,\
                        data_format=self.data_format,name=f"{name}_ds_conv1"),
                    BatchNorm2d(decay=self.decay,epsilon=self.eps,is_train=True,num_features=4*self.n_filter,act=None,data_format=self.data_format,name=f"{name}_ds_bn1")
                    ])
            self.main_block=LayerList([
                Conv2d(n_filter=self.n_filter,in_channels=self.in_channels,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format,name=f"{name}_conv1"),
                BatchNorm2d(decay=self.decay,epsilon=self.eps,is_train=True,num_features=self.n_filter,act=tf.nn.relu,data_format=self.data_format,name=f"{name}_bn1"),
                Conv2d(n_filter=self.n_filter,in_channels=self.n_filter,filter_size=(3,3),strides=self.strides,b_init=None,data_format=self.data_format,name=f"{name}_conv2"),
                BatchNorm2d(decay=self.decay,epsilon=self.eps,is_train=True,num_features=self.n_filter,act=tf.nn.relu,data_format=self.data_format,name=f"{name}_bn2"),
                Conv2d(n_filter=4*self.n_filter,in_channels=self.n_filter,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format,name=f"{name}_conv3"),
                BatchNorm2d(decay=self.decay,epsilon=self.eps,is_train=True,num_features=4*self.n_filter,act=None,data_format=self.data_format,name=f"{name}_bn3")
            ])
        
        def forward(self,x):
            res=x
            x=self.main_block.forward(x)
            if(self.downsample!=None):
                res=self.downsample.forward(res)
            return  tf.nn.relu(x+res)