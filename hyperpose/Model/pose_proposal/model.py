import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from tensorlayer.models import Model
from .define import CocoPart,CocoLimb
from ..backbones import Resnet18_backbone

class PoseProposal(Model):
    def __init__(self,parts=CocoPart,limbs=CocoLimb,colors=None,K_size=18,L_size=17,win=384,hin=384,wout=12,hout=12,wnei=9,hnei=9\
        ,lmd_rsp=0.25,lmd_iou=1,lmd_coor=5,lmd_size=5,lmd_limb=0.5,backbone=None,pretraining=False,data_format="channels_first"):
        super().__init__()
        #construct params
        self.parts=parts
        self.limbs=limbs
        self.colors=colors
        self.K=K_size
        self.L=L_size
        self.win=win
        self.hin=hin
        self.wout=wout
        self.hout=hout
        self.hnei=hnei
        self.wnei=wnei
        self.n_pos=K_size
        self.lmd_rsp=lmd_rsp
        self.lmd_iou=lmd_iou
        self.lmd_coor=lmd_coor
        self.lmd_size=lmd_size
        self.lmd_limb=lmd_limb
        self.data_format=data_format
        
        self.output_dim=6*self.K+self.hnei*self.wnei*self.L
        #construct networks
        if(backbone==None):
            self.backbone=Resnet18_backbone(scale_size=32,pretraining=pretraining,data_format=data_format)
        else:
            self.backbone=backbone(scale_size=32,pretraining=pretraining,data_format=self.data_format)
        self.add_layer_1=LayerList([
            Conv2d(n_filter=512,in_channels=self.backbone.out_channels,filter_size=(3,3),strides=(1,1),data_format=self.data_format,name="add_block_1_conv_1"),
            BatchNorm2d(decay=0.9,act=lambda x:tl.act.leaky_relu(x,alpha=0.1),is_train=True,num_features=512,data_format=self.data_format,name="add_block_1_bn_1"),
        ],name="add_block_1")
        self.add_layer_2=LayerList([
            Conv2d(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),data_format=self.data_format,name="add_block_2_conv_1"),
            BatchNorm2d(decay=0.9,act=lambda x:tl.act.leaky_relu(x,alpha=0.1),is_train=True,num_features=512,data_format=self.data_format,name="add_block_2_bn_1")
        ],name="add_block_2")
        self.add_layer_3=Conv2d(n_filter=self.output_dim,in_channels=512,filter_size=(1,1),strides=(1,1),data_format=self.data_format,name="add_block_3_conv_1")

    @tf.function
    def forward(self,x,is_train=False,domainadapt=False):
        backbone_features=self.backbone.forward(x)
        x=self.add_layer_1.forward(backbone_features)
        x=self.add_layer_2.forward(x)
        x=self.add_layer_3.forward(x)
        x=tf.nn.sigmoid(x)
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
        if(domainadapt):
            return pc,pi,px,py,pw,ph,pe,backbone_features
        else:
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
            grid_x=grid_x[:,:,np.newaxis]
            grid_y=grid_y[:,:,np.newaxis]
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
        inter_x=tf.nn.relu(tf.minimum(x1+w1/2,x2+w2/2)-tf.maximum(x1-w1/2,x2-w2/2))
        inter_y=tf.nn.relu(tf.minimum(y1+h1/2,y2+h2/2)-tf.maximum(y1-h1/2,y2-h2/2))
        inter_area=inter_x*inter_y
        union_area=area1+area2-inter_area+1e-6
        return inter_area/union_area

    def cal_loss(self,delta,tx,ty,tw,th,te,te_mask,pc,pi,px,py,pw,ph,pe,eps=1e-6):
        rtx,rty,rtw,rth=self.restore_coor(tx,ty,tw,th)
        rx,ry,rw,rh=self.restore_coor(px,py,pw,ph)
        ti=self.cal_iou((rtx,rty,rtw,rth),(rx,ry,rw,rh))
        mask_point=tf.minimum(delta+tf.where(delta<0.5,0.00001,0),1)
        mask_edge=tf.minimum(te_mask+tf.where(te_mask<0.5,0.00001,0),1)
        half=tf.where(delta<0.5,0.5,0)
        loss_rsp=self.lmd_rsp*tf.reduce_mean(tf.reduce_sum((delta-pc)**2,axis=[1,2,3]))
        loss_iou=self.lmd_iou*tf.reduce_mean(tf.reduce_sum(delta*((ti-pi)**2),axis=[1,2,3]))
        loss_coor=self.lmd_coor*tf.reduce_mean(tf.reduce_sum(mask_point*((tx-px-half)**2+(ty-py-half)**2),axis=[1,2,3]))
        loss_size=self.lmd_size*tf.reduce_mean(tf.reduce_sum(mask_point*((tf.sqrt(tw+eps)-tf.sqrt(pw+eps))**2+(tf.sqrt(th+eps)-tf.sqrt(ph+eps))**2),axis=[1,2,3]))
        loss_limb=self.lmd_limb*tf.reduce_mean(tf.reduce_sum(mask_edge*((te-pe)**2),axis=[1,2,3,4,5]))
        return loss_rsp,loss_iou,loss_coor,loss_size,loss_limb
    
    
