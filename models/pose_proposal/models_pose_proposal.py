import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from tensorlayer.models import Model

class model(Model):
    def __init__(self,K_size=18,L_size=20,win=384,hin=384,wout=12,hout=12,wnei=9,hnei=9\
        ,lmd_rsp=0.25,lmd_iou=1,lmd_coor=5,lmd_size=5,lmd_limb=0.5,back_bone='resnet_18'):
        super().__init__()
        #construct params
        self.K=K_size
        self.L=L_size
        self.win=win
        self.hin=hin
        self.wout=wout
        self.hout=hout
        self.hnei=hnei
        self.wnei=wnei
        self.lmd_rsp=lmd_rsp
        self.lmd_iou=lmd_iou
        self.lmd_coor=lmd_coor
        self.lmd_size=lmd_size
        self.lmd_limb=lmd_limb
        
        self.output_dim=6*(self.K+1)+self.hnei*self.wnei*self.L
        #construct networks
        self.back_bone=self.Resnet_18(n_filter=512,in_channels=3)
        self.add_layer_1=LayerList([
            Conv2d(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1)),
            BatchNorm2d(decay=0.9,act=lambda x:tl.act.leaky_relu(x,alpha=0.1),is_train=True,num_features=512)
        ])
        self.add_layer_2=LayerList([
            Conv2d(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1)),
            BatchNorm2d(decay=0.9,act=lambda x:tl.act.leaky_relu(x,alpha=0.1),is_train=True,num_features=512)
        ])
        self.add_layer_3=Conv2d(n_filter=self.output_dim,in_channels=512,filter_size=(1,1),strides=(1,1))

    @tf.function
    def forward(self,x,is_train=False):
        x=self.back_bone.forward(x)
        x=self.add_layer_1.forward(x)
        x=self.add_layer_2.forward(x)
        x=self.add_layer_3.forward(x)
        x=tf.transpose(x,[0,3,1,2])
        pc=x[:,0:self.K+1,:,:]
        pi=x[:,(self.K+1):2*(self.K+1),:,:]
        px=x[:,2*(self.K+1):3*(self.K+1),:,:]
        py=x[:,3*(self.K+1):4*(self.K+1),:,:]
        pw=tf.nn.sigmoid(x[:,4*(self.K+1):5*(self.K+1),:,:])
        ph=tf.nn.sigmoid(x[:,5*(self.K+1):6*(self.K+1),:,:])
        pe=tf.reshape(x[:,6*(self.K+1):,:,:],[-1,self.L,self.wnei,self.hnei,self.wout,self.hout])
        return pc,pi,px,py,pw,ph,pe
    
    def restore_coor(self,x,y,w,h):
        grid_size_x=self.win/self.wout
        grid_size_y=self.hin/self.hout
        grid_x,grid_y=tf.meshgrid(np.arange(self.wout).astype(np.float32),np.arange(self.hout).astype(np.float32))
        rx=(x+grid_x)*grid_size_x
        ry=(y+grid_y)*grid_size_y
        rw=w*self.win
        rh=h*self.hin
        return rx,ry,rw,rh
    
    def cal_iou(self,bbx1,bbx2):
        #input x,y are the center of bbx
        x1,y1,w1,h1=bbx1
        x2,y2,w2,h2=bbx2
        area1=w1*h1
        area2=w2*h2
        inter_x=tf.nn.relu(tf.minimum(x1+w1//2,x2+w2//2)-tf.maximum(x1-w1//2,x2-w2//2))
        inter_y=tf.nn.relu(tf.minimum(y1+h1//2,y2+h2//2)-tf.maximum(y1-h1//2,y2-h2//2))
        inter_area=inter_x*inter_y
        union_area=area1+area2-inter_area
        return inter_area/union_area

    def cal_loss(self,delta,tx,ty,tw,th,te,te_mask,pc,pi,px,py,pw,ph,pe):
        rtx,rty,rtw,rth=self.restore_coor(tx,ty,tw,th)
        rx,ry,rw,rh=self.restore_coor(px,py,pw,ph)
        ti=self.cal_iou((rtx,rty,rtw,rth),(rx,ry,rw,rh))
        loss_rsp=self.lmd_rsp*tf.reduce_mean(tf.reduce_sum((delta-pc)**2,axis=[1,2,3]))
        loss_iou=self.lmd_iou*tf.reduce_mean(tf.reduce_sum(delta*((ti-pi)**2),axis=[1,2,3]))
        loss_coor=self.lmd_coor*tf.reduce_mean(tf.reduce_sum(delta*((tx-px)**2+(ty-py)**2),axis=[1,2,3]))
        loss_size=self.lmd_size*tf.reduce_mean(tf.reduce_sum(delta*((tw**0.5-pw**0.5)**2+(th**0.5-ph**0.5)**2),axis=[1,2,3]))
        loss_limb=self.lmd_limb*tf.reduce_mean(tf.reduce_sum(te_mask*((te-pe)**2),axis=[1,2,3]))
        return loss_rsp,loss_iou,loss_coor,loss_size,loss_limb
    
    class Resnet_18(Model):
        def __init__(self,n_filter=512,in_channels=3):
            super().__init__()
            self.conv1=Conv2d(n_filter=64,in_channels=in_channels,filter_size=(7,7),strides=(2,2),b_init=None)
            self.bn1=BatchNorm2d(decay=0.9,act=tf.nn.relu,is_train=True,num_features=64)
            self.maxpool=MaxPool2d(filter_size=(3,3),strides=(2,2))
            self.res_block_2_1=self.Res_block(n_filter=64,in_channels=64,strides=(1,1),is_down_sample=False)
            self.res_block_2_2=self.Res_block(n_filter=64,in_channels=64,strides=(1,1),is_down_sample=False)
            self.res_block_3_1=self.Res_block(n_filter=128,in_channels=64,strides=(2,2),is_down_sample=True)
            self.res_block_3_2=self.Res_block(n_filter=128,in_channels=128,strides=(1,1),is_down_sample=False)
            self.res_block_4_1=self.Res_block(n_filter=256,in_channels=128,strides=(2,2),is_down_sample=True)
            self.res_block_4_2=self.Res_block(n_filter=256,in_channels=256,strides=(1,1),is_down_sample=False)
            self.res_block_5_1=self.Res_block(n_filter=n_filter,in_channels=256,strides=(2,2),is_down_sample=True)
        
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
            return x

        class Res_block(Model):
            def __init__(self,n_filter,in_channels,strides=(1,1),is_down_sample=False):
                super().__init__()
                self.is_down_sample=is_down_sample
                self.main_block=LayerList([
                Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(3,3),strides=strides,b_init=None),
                BatchNorm2d(decay=0.9,act=tf.nn.relu,is_train=True,num_features=n_filter),
                Conv2d(n_filter=n_filter,in_channels=n_filter,filter_size=(3,3),strides=(1,1),b_init=None),
                BatchNorm2d(decay=0.9,is_train=True,num_features=n_filter),
                ])
                if(self.is_down_sample):
                    self.down_sample=LayerList([
                        Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(3,3),strides=strides,b_init=None),
                        BatchNorm2d(decay=0.9,is_train=True,num_features=n_filter)
                    ])

            def forward(self,x):
                res=x
                x=self.main_block.forward(x)
                if(self.is_down_sample):
                    res=self.down_sample.forward(res)
                return tf.nn.relu(x+res)
