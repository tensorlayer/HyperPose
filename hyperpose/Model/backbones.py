import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d, SeparableConv2d

class MobilenetV1_backbone(Model):
    def __init__(self,scale_size=8,data_format="channel_last"):
        super().__init__()
        self.data_format=data_format
        self.scale_size=scale_size
        if(self.scale_size==8):
            strides=(1,1)
        else:
            strides=(2,2)
        self.out_channels=1024
        self.layer_list=[]
        self.layer_list+=self.conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(2,2))
        self.layer_list+=self.separable_conv_block(n_filter=64,in_channels=32,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(2,2))
        self.layer_list+=self.separable_conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(2,2))
        self.layer_list+=self.separable_conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=strides)
        self.layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1))
        self.layer_list+=self.separable_conv_block(n_filter=1024,in_channels=512,filter_size=(3,3),strides=strides)
        self.layer_list+=self.separable_conv_block(n_filter=1024,in_channels=1024,filter_size=(3,3),strides=(1,1))
        self.main_block=LayerList(self.layer_list)
    
    def forward(self,x):
        return self.main_block.forward(x)

    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),padding="SAME"):
        layer_list=[]
        layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            data_format=self.data_format,padding=padding))
        layer_list.append(BatchNorm2d(num_features=n_filter,is_train=True,act=tf.nn.relu,data_format=self.data_format))
        return LayerList(layer_list)
    
    def separable_conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1)):
        layer_list=[]
        layer_list.append(DepthwiseConv2d(in_channels=in_channels,filter_size=filter_size,strides=strides,data_format=self.data_format))
        layer_list.append(BatchNorm2d(num_features=in_channels,is_train=True,act=tf.nn.relu,data_format=self.data_format))
        layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(1,1),strides=(1,1),data_format=self.data_format))
        layer_list.append(BatchNorm2d(num_features=n_filter,is_train=True,act=tf.nn.relu,data_format=self.data_format))
        return LayerList(layer_list)

class MobilenetV2_backbone(Model):
    def __init__(self,scale_size=8,data_format="channels_last"):
        super().__init__()
        self.data_format=data_format
        self.scale_size=scale_size
        self.out_channels=320
        if(self.scale_size==8):
            strides=(1,1)
        else:
            strides=(2,2)
        #block_1 n=1
        self.block_1_1=Conv2d(n_filter=32,in_channels=3,filter_size=(3,3),strides=(2,2),data_format=self.data_format)
        self.block_1_2=BatchNorm2d(num_features=32,is_train=True,act=tf.nn.relu6,data_format=self.data_format)
        #block_2 n=1
        self.block_2_1=self.InvertedResidual(n_filter=16,in_channels=32,strides=(1,1),exp_ratio=1,data_format=self.data_format)
        #block_3 n=2
        self.block_3_1=self.InvertedResidual(n_filter=24,in_channels=16,strides=(2,2),exp_ratio=6,data_format=self.data_format)
        self.block_3_2=self.InvertedResidual(n_filter=24,in_channels=24,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        #block_4 n=3
        self.block_4_1=self.InvertedResidual(n_filter=32,in_channels=24,strides=(2,2),exp_ratio=6,data_format=self.data_format)
        self.block_4_2=self.InvertedResidual(n_filter=32,in_channels=32,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        self.block_4_3=self.InvertedResidual(n_filter=32,in_channels=32,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        #block_5 n=4
        self.block_5_1=self.InvertedResidual(n_filter=64,in_channels=32,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        self.block_5_2=self.InvertedResidual(n_filter=64,in_channels=64,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        self.block_5_3=self.InvertedResidual(n_filter=64,in_channels=64,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        self.block_5_4=self.InvertedResidual(n_filter=64,in_channels=64,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        #block_6 n=3
        self.block_6_1=self.InvertedResidual(n_filter=96,in_channels=64,strides=strides,exp_ratio=6,data_format=self.data_format)
        self.block_6_2=self.InvertedResidual(n_filter=96,in_channels=96,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        self.block_6_3=self.InvertedResidual(n_filter=96,in_channels=96,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        #block_7 n=3
        self.block_7_1=self.InvertedResidual(n_filter=160,in_channels=96,strides=strides,exp_ratio=6,data_format=self.data_format)
        self.block_7_2=self.InvertedResidual(n_filter=160,in_channels=160,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        self.block_7_3=self.InvertedResidual(n_filter=160,in_channels=160,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        #block_8 n=1
        self.block_8=self.InvertedResidual(n_filter=320,in_channels=160,strides=(1,1),exp_ratio=6,data_format=self.data_format)
        
    def forward(self,x):
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
        x=self.block_6_1.forward(x)
        x=self.block_6_2.forward(x)
        x=self.block_6_3.forward(x)
        x=self.block_7_1.forward(x)
        x=self.block_7_2.forward(x)
        x=self.block_7_3.forward(x)
        x=self.block_8.forward(x)
        return x

    class InvertedResidual(Model):
        def __init__(self,n_filter=128,in_channels=128,strides=(1,1),exp_ratio=6,data_format="channels_first"):
            super().__init__()
            self.n_filter=n_filter
            self.in_channels=in_channels
            self.strides=strides
            self.exp_ratio=exp_ratio
            self.data_format=data_format
            self.hidden_dim=self.exp_ratio*self.in_channels
            self.identity=False
            if(self.strides==(1,1) and self.in_channels==self.n_filter):
                self.identity=True
            if(self.exp_ratio==1):
                self.main_block=LayerList([
                    DepthwiseConv2d(in_channels=self.hidden_dim,filter_size=(3,3),strides=self.strides,\
                        b_init=None,data_format=self.data_format),
                    BatchNorm2d(num_features=self.hidden_dim,is_train=True,act=tf.nn.relu6,data_format=self.data_format),
                    Conv2d(n_filter=self.n_filter,in_channels=self.hidden_dim,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format),
                    BatchNorm2d(num_features=self.n_filter,is_train=True,act=None,data_format=self.data_format)
                ])
            else:
                self.main_block=LayerList([
                    Conv2d(n_filter=self.hidden_dim,in_channels=self.in_channels,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format),
                    BatchNorm2d(num_features=self.hidden_dim,is_train=True,act=tf.nn.relu6,data_format=self.data_format),
                    DepthwiseConv2d(in_channels=self.hidden_dim,filter_size=(3,3),strides=self.strides,\
                        b_init=None,data_format=self.data_format),
                    BatchNorm2d(num_features=self.hidden_dim,is_train=True,act=tf.nn.relu6,data_format=self.data_format),
                    Conv2d(n_filter=self.n_filter,in_channels=self.hidden_dim,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format)
                ])

        def forward(self,x):
            if(self.identity):
                return x+self.main_block.forward(x)
            else:
                return self.main_block.forward(x)

class vggtiny_backbone(Model):
    def __init__(self,in_channels=3,scale_size=8,data_format="channels_first"):
        super().__init__()
        self.in_channels=in_channels
        self.data_format=data_format
        self.scale_size=scale_size
        if(self.scale_size==8):
            strides=(1,1)
        else:
            strides=(2,2)
        self.out_channels=384
        self.main_block=layers.LayerList([
            self.conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1)),
            self.conv_block(n_filter=64,in_channels=32,filter_size=(3,3),strides=(1,1)),
            MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format),
            self.conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(1,1)),
            self.conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1)),
            MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format),
            self.conv_block(n_filter=200,in_channels=128,filter_size=(3,3),strides=(1,1)),
            self.conv_block(n_filter=200,in_channels=200,filter_size=(3,3),strides=strides),
            self.conv_block(n_filter=200,in_channels=200,filter_size=(3,3),strides=(1,1)),
            MaxPool2d(filter_size=(2,2),strides=(2,2),padding="SAME",data_format=self.data_format),
            self.conv_block(n_filter=384,in_channels=200,filter_size=(3,3),strides=(1,1)),
            self.conv_block(n_filter=384,in_channels=384,filter_size=(3,3),strides=strides)
        ])
    
    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,padding="SAME"):
        layer_list=[]
        layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            act=None,data_format=self.data_format,padding=padding))
        layer_list.append(BatchNorm2d(num_features=n_filter,act=act,is_train=True,data_format=self.data_format))
        return LayerList(layer_list)
    
    def forward(self,x):
        return self.main_block.forward(x)

class vgg16_backbone(Model):
    def __init__(self,in_channels=3,scale_size=8,data_format="channels_first"):
        super().__init__()
        self.in_channels=in_channels
        self.data_format=data_format
        self.scale_size=scale_size
        if(self.scale_size==8):
            strides=(1,1)
        elif(self.scale_size==32):
            strides=(2,2)
        self.out_channels=512
        self.layer_list=[
            self.conv_block(n_filter=64,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=64,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
            self.conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
            self.conv_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
            self.conv_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu)
        ]
        if(self.scale_size==32):
            self.layer_list+=[MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format)]
        self.layer_list+=[
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=strides,act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu)
        ]
        self.main_block=LayerList(self.layer_list)

    def forward(self,x):
        return self.main_block.forward(x)
    
    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=None,padding="SAME"):
        return Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            act=act,data_format=self.data_format,padding=padding)

class vgg19_backbone(Model):
    def __init__(self,in_channels=3,scale_size=8,data_format="channels_first"):
        super().__init__()
        self.in_channels=in_channels
        self.data_format=data_format
        self.scale_size=scale_size
        if(self.scale_size==8):
            strides=(1,1)
        else:
            strides=(2,2)
        self.out_channels=512
        self.layer_list=[
            self.conv_block(n_filter=64,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=64,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
            self.conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
            self.conv_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
            self.conv_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu)
        ]
        if(self.scale_size==32):
            self.layer_list+=[MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format)]
        self.layer_list+=[
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=strides,act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
            self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu)
        ]
        self.main_block=LayerList(self.layer_list)
    
    
    def forward(self,x):
        return self.main_block.forward(x)
    
    def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=None,padding="SAME"):
        return Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
            act=act,data_format=self.data_format,padding=padding)


class Resnet18_backbone(Model):
    def __init__(self,in_channels=3,n_filter=64,scale_size=8,data_format="channels_first"):
        super().__init__()
        self.in_channels=in_channels
        self.n_filter=n_filter
        self.scale_size=scale_size
        self.data_format=data_format
        self.out_channels=512
        self.conv1=Conv2d(n_filter=64,in_channels=self.in_channels,filter_size=(7,7),strides=(2,2),padding="SAME",b_init=None,data_format=self.data_format)
        self.bn1=BatchNorm2d(is_train=True,num_features=64,data_format=self.data_format,act=tf.nn.relu)
        self.maxpool1=MaxPool2d(filter_size=(3,3),strides=(2,2),data_format=self.data_format)
        self.layer1=self.Basic_block(in_channels=64,n_filter=64,strides=(1,1),data_format=self.data_format)
        self.layer2=self.Basic_block(in_channels=64,n_filter=128,strides=(2,2),data_format=self.data_format)
        if(self.scale_size==8):
            strides=(1,1)
        else:
            strides=(2,2)
        self.layer3=self.Basic_block(in_channels=128,n_filter=256,strides=strides,data_format=self.data_format)
        self.layer4=self.Basic_block(in_channels=256,n_filter=512,strides=strides,data_format=self.data_format)
    
    def forward(self,x):
        x=self.conv1.forward(x)
        x=self.bn1.forward(x)
        x=self.maxpool1.forward(x)
        x=self.layer1.forward(x)
        x=self.layer2.forward(x)
        x=self.layer3.forward(x)
        x=self.layer4.forward(x)
        return x
    
    class Basic_block(Model):
        def __init__(self,in_channels=64,n_filter=64,strides=(1,1),data_format="channels_first"):
            super().__init__()
            self.in_channels=in_channels
            self.n_filter=n_filter
            self.strides=strides
            self.data_format=data_format
            self.downsample=None
            if(self.strides!=(1,1) or self.in_channels!=self.n_filter):
                self.downsample=LayerList([
                    Conv2d(n_filter=self.n_filter,in_channels=self.in_channels,filter_size=(1,1),strides=self.strides,b_init=None,\
                        data_format=self.data_format),
                    BatchNorm2d(is_train=True,num_features=self.n_filter,data_format=self.data_format)
                    ])
            self.main_block=LayerList([
                Conv2d(n_filter=self.n_filter,in_channels=self.in_channels,filter_size=(3,3),strides=self.strides,b_init=None,data_format=self.data_format),
                BatchNorm2d(is_train=True,num_features=self.n_filter,act=tf.nn.relu,data_format=self.data_format),
                Conv2d(n_filter=self.n_filter,in_channels=self.n_filter,filter_size=(3,3),b_init=None,data_format=self.data_format),
                BatchNorm2d(is_train=True,num_features=self.n_filter,data_format=self.data_format)
            ])
        
        def forward(self,x):
            res=x
            x=self.main_block.forward(x)
            if(self.downsample!=None):
                res=self.downsample.forward(res)
            return tf.nn.relu(res+x)    

class Resnet50_backbone(Model):
    def __init__(self,in_channels=3,n_filter=64,scale_size=8,data_format="channels_first"):
        super().__init__()
        self.in_channels=in_channels
        self.n_filter=n_filter
        self.scale_size=scale_size
        self.data_format=data_format
        self.out_channels=2048
        self.conv1=Conv2d(n_filter=64,in_channels=self.in_channels,filter_size=(7,7),strides=(2,2),padding="SAME",b_init=None,data_format=self.data_format)
        self.bn1=BatchNorm2d(is_train=True,num_features=64,data_format=self.data_format,act=tf.nn.relu)
        self.maxpool1=MaxPool2d(filter_size=(3,3),strides=(2,2),data_format=self.data_format)
        self.layer1=self.Basic_block(in_channels=64,n_filter=64,strides=(1,1),data_format=self.data_format)
        self.layer2=self.Basic_block(in_channels=256,n_filter=128,strides=(2,2),data_format=self.data_format)
        if(self.scale_size==8):
            strides=(1,1)
        else:
            strides=(2,2)
        self.layer3=self.Basic_block(in_channels=512,n_filter=256,strides=strides,data_format=self.data_format)
        self.layer4=self.Basic_block(in_channels=1024,n_filter=512,strides=strides,data_format=self.data_format)
    
    def forward(self,x):
        x=self.conv1.forward(x)
        x=self.bn1.forward(x)
        x=self.maxpool1.forward(x)
        x=self.layer1.forward(x)
        x=self.layer2.forward(x)
        x=self.layer3.forward(x)
        x=self.layer4.forward(x)
        return x

    class Basic_block(Model):
        def __init__(self,in_channels=64,n_filter=64,strides=(1,1),data_format="channels_first"):
            super().__init__()
            self.in_channels=in_channels
            self.n_filter=n_filter
            self.strides=strides
            self.data_format=data_format
            self.downsample=None
            if(self.strides!=(1,1) or self.in_channels!=4*self.n_filter):
                self.downsample=LayerList([
                    Conv2d(n_filter=4*self.n_filter,in_channels=self.in_channels,filter_size=(1,1),strides=self.strides,b_init=None,\
                        data_format=self.data_format),
                    BatchNorm2d(is_train=True,num_features=4*self.n_filter,data_format=self.data_format)
                    ])
            self.main_block=LayerList([
                Conv2d(n_filter=self.n_filter,in_channels=self.in_channels,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format),
                BatchNorm2d(is_train=True,num_features=self.n_filter,act=tf.nn.relu,data_format=self.data_format),
                Conv2d(n_filter=self.n_filter,in_channels=self.n_filter,filter_size=(3,3),strides=self.strides,b_init=None,data_format=self.data_format),
                BatchNorm2d(is_train=True,num_features=self.n_filter,act=tf.nn.relu,data_format=self.data_format),
                Conv2d(n_filter=4*self.n_filter,in_channels=self.n_filter,filter_size=(1,1),strides=(1,1),b_init=None,data_format=self.data_format),
                BatchNorm2d(is_train=True,num_features=4*self.n_filter,data_format=self.data_format,)
            ])
        
        def forward(self,x):
            res=x
            x=self.main_block.forward(x)
            if(self.downsample!=None):
                res=self.downsample.forward(res)
            return  tf.nn.relu(x+res)