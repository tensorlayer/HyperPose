import tensorflow as tf
import tensorlayer as tl
from tensorlayer.models import Model
from tensorlayer.layers import Conv2d,BatchNorm2d,Dense,Flatten,LayerList

def get_discriminator(feature_hin,feature_win,in_channnels,n_filter=512,layer_num=5,data_format="channels_first"):
    last_channels=in_channnels
    layer_list=[]
    return layer_list


class Discriminator(Model):
    def __init__(self,feature_hin,feature_win,in_channels,data_format="channels_first"):
        super().__init__()
        self.data_format=data_format
        self.feature_hin=feature_hin
        self.feature_win=feature_win
        self.in_channels=in_channels
        self.layer_num=5
        self.n_filter=256
        # construct Model
        layer_list=[]
        last_channels=self.in_channels
        dis_hin,dis_win=self.feature_hin,self.feature_win
        for layer_idx in range(0,self.layer_num):
            strides=(1,1)
            if(dis_hin>=4 or dis_win>=4):
                strides=(2,2)
                dis_hin,dis_win=(dis_hin+1)//2,(dis_win+1)//2
            layer_list+=[
                Conv2d(n_filter=self.n_filter,in_channels=last_channels,strides=strides,act=tf.nn.relu,\
                    data_format=data_format,name=f"dis_conv_{layer_idx}")
            ]
            last_channels=self.n_filter
        layer_list.append(Flatten(name="Flatten"))
        layer_list.append(Dense(n_units=4096,in_channels=dis_hin*dis_win*self.n_filter,act=tf.nn.relu,name="fc1"))
        layer_list.append(Dense(n_units=1000,in_channels=4096,act=tf.nn.relu,name="fc2"))
        layer_list.append(Dense(n_units=1,in_channels=1000,act=None,name="fc3"))
        self.main_block=LayerList(layer_list)
    
    def forward(self,x):
        return self.main_block.forward(x)
    
    def cal_loss(self,x,label):
        label_shape = [x.shape[0],1]
        if(label == True):
            gt_label = tf.ones(shape=label_shape)
        elif(label == False):
            gt_label = tf.zeros(shape=label_shape)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_label,logits=x))
        return loss
    
    
    