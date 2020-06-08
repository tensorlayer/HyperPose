import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from tensorlayer.models import Model

class PoseProposal(Model):
    def __init__(self,K_size=18,L_size=17,win=384,hin=384,wout=12,hout=12,wnei=9,hnei=9\
        ,lmd_rsp=0.25,lmd_iou=1,lmd_coor=5,lmd_size=5,lmd_limb=0.5,backbone=None,data_format="channels_first"):
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
        self.data_format=data_format
        
        self.output_dim=6*self.K+self.hnei*self.wnei*self.L
        #construct networks
        if(backbone==None):
            self.backbone=self.Resnet_18(n_filter=512,in_channels=3,data_format=data_format)
        else:
            self.backbone=backbone(scale_size=32,data_format=self.data_format)
        self.add_layer_1=LayerList([
            Conv2d(n_filter=512,in_channels=self.backbone.out_channels,filter_size=(3,3),strides=(1,1),data_format=self.data_format),
            BatchNorm2d(decay=0.9,act=lambda x:tl.act.leaky_relu(x,alpha=0.1),is_train=True,num_features=512,data_format=self.data_format)
        ])
        self.add_layer_2=LayerList([
            Conv2d(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),data_format=self.data_format),
            BatchNorm2d(decay=0.9,act=lambda x:tl.act.leaky_relu(x,alpha=0.1),is_train=True,num_features=512,data_format=self.data_format)
        ])
        self.add_layer_3=Conv2d(n_filter=self.output_dim,in_channels=512,filter_size=(1,1),strides=(1,1),data_format=self.data_format)

    @tf.function
    def forward(self,x,is_train=False):
        x=self.backbone.forward(x)
        x=self.add_layer_1.forward(x)
        x=self.add_layer_2.forward(x)
        x=self.add_layer_3.forward(x)
        if(self.data_format=="channels_first"):
            pc=x[:,0:self.K,:,:]
            pi=x[:,self.K:2*self.K,:,:]
            px=x[:,2*self.K:3*self.K,:,:]
            py=x[:,3*self.K:4*self.K,:,:]
            pw=x[:,4*self.K:5*self.K,:,:]
            ph=x[:,5*self.K:6*self.K,:,:]
            pe=tf.reshape(x[:,6*self.K:,:,:],[-1,self.L,self.wnei,self.hnei,self.wout,self.hout])
        else:
            pc=x[:,:,:,0:self.K]
            pi=x[:,:,:,self.K:2*self.K]
            px=x[:,:,:,2*self.K:3*self.K]
            py=x[:,:,:,3*self.K:4*self.K]
            pw=x[:,:,:,4*self.K:5*self.K]
            ph=x[:,:,:,5*self.K:6*self.K]
            pe=tf.reshape(x[:,:,:,6*self.K:],[-1,self.wnei,self.hnei,self.wout,self.hout,self.L])
        if(is_train==False):
            px,py,pw,ph=self.restore_coor(px,py,pw,ph)
        return pc,pi,px,py,pw,ph,pe
    
    @tf.function
    def infer(self,x):
        pc,pi,px,py,pw,ph,pe=self.forward(x,is_train=False)
        return pc,pi,px,py,pw,ph,pe
    
    def restore_coor(self,x,y,w,h):
        grid_size_x=self.win/self.wout
        grid_size_y=self.hin/self.hout
        grid_x,grid_y=tf.meshgrid(np.arange(self.wout).astype(np.float32),np.arange(self.hout).astype(np.float32))
        if(self.data_format=="channels_last"):
            grid_size_x=grid_size_x[:,:,np.newaxis]
            grid_size_y=grid_size_y[:,:,np.newaxis]
        rx=(x+grid_x)*grid_size_x
        ry=(y+grid_y)*grid_size_y
        rw=(w**2)*self.win
        rh=(h**2)*self.hin
        return rx,ry,rw,rh
    
    def cal_iou(self,bbx1,bbx2):
        #input x,y are the center of bbx
        x1,y1,w1,h1=bbx1
        x2,y2,w2,h2=bbx2
        area1=w1*h1
        area2=w2*h2
        inter_x=tf.nn.relu(tf.minimum(x1+w1/2,x2+w2/2)-tf.maximum(x1-w1/2,x2-w2/2))
        inter_y=tf.nn.relu(tf.minimum(y1+h1/2,y2+h2/2)-tf.maximum(y1-h1/2,y2-h2/2))
        inter_area=inter_x*inter_y
        union_area=area1+area2-inter_area+1e-6
        return inter_area/union_area

    def cal_loss(self,delta,tx,ty,tw,th,te,te_mask,pc,pi,px,py,pw,ph,pe):
        rtx,rty,rtw,rth=self.restore_coor(tx,ty,tw,th)
        rx,ry,rw,rh=self.restore_coor(px,py,pw,ph)
        ti=self.cal_iou((rtx,rty,rtw,rth),(rx,ry,rw,rh))
        loss_rsp=self.lmd_rsp*tf.reduce_mean(tf.reduce_sum((delta-pc)**2,axis=[1,2,3]))
        loss_iou=self.lmd_iou*tf.reduce_mean(tf.reduce_sum(delta*((ti-pi)**2),axis=[1,2,3]))
        loss_coor=self.lmd_coor*tf.reduce_mean(tf.reduce_sum(delta*((tx-px)**2+(ty-py)**2),axis=[1,2,3]))
        loss_size=self.lmd_size*tf.reduce_mean(tf.reduce_sum(delta*((tw-pw)**2+(th-ph)**2),axis=[1,2,3]))
        loss_limb=self.lmd_limb*tf.reduce_mean(tf.reduce_sum(te_mask*((te-pe)**2),axis=[1,2,3,4,5]))
        return loss_rsp,loss_iou,loss_coor,loss_size,loss_limb
    
    class Resnet_18(Model):
        def __init__(self,n_filter=512,in_channels=3,data_format="channels_first"):
            super().__init__()
            self.data_format=data_format
            self.out_channels=n_filter
            self.conv1=Conv2d(n_filter=64,in_channels=in_channels,filter_size=(7,7),strides=(2,2),b_init=None,data_format=self.data_format)
            self.bn1=BatchNorm2d(decay=0.9,act=tf.nn.relu,is_train=True,num_features=64,data_format=self.data_format)
            self.maxpool=MaxPool2d(filter_size=(3,3),strides=(2,2),data_format=self.data_format)
            self.res_block_2_1=self.Res_block(n_filter=64,in_channels=64,strides=(1,1),is_down_sample=False,data_format=self.data_format)
            self.res_block_2_2=self.Res_block(n_filter=64,in_channels=64,strides=(1,1),is_down_sample=False,data_format=self.data_format)
            self.res_block_3_1=self.Res_block(n_filter=128,in_channels=64,strides=(2,2),is_down_sample=True,data_format=self.data_format)
            self.res_block_3_2=self.Res_block(n_filter=128,in_channels=128,strides=(1,1),is_down_sample=False,data_format=self.data_format)
            self.res_block_4_1=self.Res_block(n_filter=256,in_channels=128,strides=(2,2),is_down_sample=True,data_format=self.data_format)
            self.res_block_4_2=self.Res_block(n_filter=256,in_channels=256,strides=(1,1),is_down_sample=False,data_format=self.data_format)
            self.res_block_5_1=self.Res_block(n_filter=n_filter,in_channels=256,strides=(2,2),is_down_sample=True,data_format=self.data_format)
        
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
            def __init__(self,n_filter,in_channels,strides=(1,1),is_down_sample=False,data_format="channels_first"):
                super().__init__()
                self.data_format=data_format
                self.is_down_sample=is_down_sample
                self.main_block=LayerList([
                Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(3,3),strides=strides,b_init=None,data_format=self.data_format),
                BatchNorm2d(decay=0.9,act=tf.nn.relu,is_train=True,num_features=n_filter,data_format=self.data_format),
                Conv2d(n_filter=n_filter,in_channels=n_filter,filter_size=(3,3),strides=(1,1),b_init=None,data_format=self.data_format),
                BatchNorm2d(decay=0.9,is_train=True,num_features=n_filter,data_format=self.data_format),
                ])
                if(self.is_down_sample):
                    self.down_sample=LayerList([
                        Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(3,3),strides=strides,b_init=None,data_format=self.data_format),
                        BatchNorm2d(decay=0.9,is_train=True,num_features=n_filter,data_format=self.data_format)
                    ])

            def forward(self,x):
                res=x
                x=self.main_block.forward(x)
                if(self.is_down_sample):
                    res=self.down_sample.forward(res)
                return tf.nn.relu(x+res)
