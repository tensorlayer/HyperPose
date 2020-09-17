import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
from tensorlayer.layers import Conv2d,BatchNorm2d,Dense,Flatten,LayerList

def get_discriminator(train_model,n_filter=512,layer_num=5):
    scale_size=train_model.backbone.scale_size
    data_format=train_model.data_format
    feature_hin,feature_win=train_model.hin/scale_size,train_model.win/scale_size
    dis_hin,dis_win=feature_hin,feature_win
    last_channels=train_model.backbone.out_channels
    layer_list=[]
    for layer_idx in range(0,layer_num):
        strides=(1,1)
        if(dis_hin>10 or dis_win>10):
            strides=(2,2)
            dis_hin,dis_win=(dis_hin+1)//2,(dis_win+1)//2
        layer_list+=[
            Conv2d(n_filter=n_filter,in_channels=last_channels,strides=strides,act=tf.nn.relu,data_format=data_format,\
                name=f"dis_conv_{layer_idx}")
        ]
        last_channels=n_filter
    layer_list.append(Flatten(name="Flatten"))
    layer_list.append(Dense(n_units=4096,in_channels=dis_hin*dis_win*n_filter,act=tf.nn.relu,name="fc1"))
    layer_list.append(Dense(n_units=1000,in_channels=4096,act=tf.nn.relu,name="fc2"))
    layer_list.append(Dense(n_units=1,in_channels=1000,act=None,name="fc3"))
    discriminator=Discriminator(layer_list=layer_list,data_format=data_format)
    print("domain adaptation discriminator generated!")
    return discriminator


class Discriminator(Model):
    def __init__(self,layer_list,data_format="channels_first"):
        self.data_format=data_format
        self.main_block=LayerList(layer_list)
    
    def forward(self,x):
        return self.main_block.forward(x)
    
    def cal_loss(self,x,labels):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=x)
    
    
    