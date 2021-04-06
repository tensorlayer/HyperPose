import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from .define import CocoColor
from ..backbones import Resnet50_backbone


class Pifpaf(Model):
    def __init__(self,parts,limbs,colors=CocoColor,n_pos=17,n_limbs=19,hin=368,win=368,scale_size=32,backbone=None,pretraining=False,quad_size=2,quad_num=1,
    lambda_pif_conf=1.0,lambda_pif_vec=1.0,lambda_pif_scale=1.0,lambda_paf_conf=1.0,lambda_paf_src_vec=1.0,lambda_paf_dst_vec=1.0,
    lambda_paf_src_scale=1.0,lambda_paf_dst_scale=1.0,data_format="channels_first"):
        super().__init__()
        self.parts=parts
        self.limbs=limbs
        self.n_pos=n_pos
        self.n_limbs=n_limbs
        self.colors=colors
        self.hin=hin
        self.win=win
        self.quad_size=quad_size
        self.quad_num=quad_num
        self.scale_size=scale_size
        self.stride=int(self.scale_size/(self.quad_size**self.quad_num))
        self.lambda_pif_conf=lambda_pif_conf
        self.lambda_pif_vec=lambda_pif_vec
        self.lambda_pif_scale=lambda_pif_scale
        self.lambda_paf_conf=lambda_paf_conf
        self.lambda_paf_src_vec=lambda_paf_src_vec
        self.lambda_paf_dst_vec=lambda_paf_dst_vec
        self.lambda_paf_src_scale=lambda_paf_src_scale
        self.lambda_paf_dst_scale=lambda_paf_dst_scale
        self.data_format=data_format
        if(backbone==None):
            self.backbone=Resnet50_backbone(data_format=data_format,use_pool=False,scale_size=self.scale_size,decay=0.99,eps=1e-4)
            self.stride=int(self.stride/2) #because of not using max_pool layer of resnet50
        else:
            self.backbone=backbone(data_format=data_format,scale_size=self.scale_size)
        self.hout=int(hin/self.stride)
        self.wout=int(win/self.stride)
        #generate mesh grid
        x_range=np.linspace(start=0,stop=self.wout-1,num=self.wout)
        y_range=np.linspace(start=0,stop=self.hout-1,num=self.hout)
        mesh_x,mesh_y=np.meshgrid(x_range,y_range)
        self.mesh_grid=np.stack([mesh_x,mesh_y])
        #construct head
        self.pif_head=self.PifHead(input_features=self.backbone.out_channels,n_pos=self.n_pos,n_limbs=self.n_limbs,\
            quad_size=self.quad_size,hout=self.hout,wout=self.wout,stride=self.stride,mesh_grid=self.mesh_grid,data_format=self.data_format)
        self.paf_head=self.PafHead(input_features=self.backbone.out_channels,n_pos=self.n_pos,n_limbs=self.n_limbs,\
            quad_size=self.quad_size,hout=self.hout,wout=self.wout,stride=self.stride,mesh_grid=self.mesh_grid,data_format=self.data_format)
    
    @tf.function(experimental_relax_shapes=True)
    def forward(self,x,is_train=False):
        x=self.backbone.forward(x)
        pif_maps=self.pif_head.forward(x,is_train=is_train)
        paf_maps=self.paf_head.forward(x,is_train=is_train)
        return pif_maps,paf_maps
    
    @tf.function(experimental_relax_shapes=True)
    def infer(self,x):
        pif_maps,paf_maps=self.forward(x,is_train=False)
        pif_conf,pif_vec,_,pif_scale=pif_maps
        paf_conf,paf_src_vec,paf_dst_vec,_,_,paf_src_scale,paf_dst_scale=paf_maps
        return pif_conf,pif_vec,pif_scale,paf_conf,paf_src_vec,paf_dst_vec,paf_src_scale,paf_dst_scale
    
    def soft_clamp(self,x,max_value=5.0):
        above_mask=tf.where(x>=max_value,1.0,0.0)
        x_below=x*(1-above_mask)
        x_soft_above=tf.where(x>=max_value,x,max_value)
        x_above=(max_value+tf.math.log(1+x_soft_above-max_value))*above_mask
        return x_below+x_above
    
    def Bce_loss(self,pd_conf,gt_conf,focal_gamma=1.0):
        #shape conf:[batch,field,h,w]
        batch_size=pd_conf.shape[0]
        valid_mask=tf.logical_not(tf.math.is_nan(gt_conf))
        #select pd_conf
        pd_conf=pd_conf[valid_mask]
        #select gt_conf
        gt_conf=gt_conf[valid_mask]
        #calculate loss
        bce_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=pd_conf,labels=gt_conf)
        bce_loss=self.soft_clamp(bce_loss)
        if(focal_gamma!=0.0):
            p=tf.nn.sigmoid(pd_conf)
            pt=p*gt_conf+(1-p)*(1-gt_conf)
            focal=1.0-pt
            if(focal_gamma!=1.0):
                focal=(focal+1e-4)**focal_gamma
            bce_loss=focal*bce_loss*0.5
        bce_loss=tf.reduce_sum(bce_loss)/batch_size
        return bce_loss
    
    def Laplace_loss(self,pd_vec,pd_logb,gt_vec,gt_bmin):
        #shape vec: [batch,field,2,h,w]
        #shape logb: [batch,field,h,w]
        batch_size=pd_vec.shape[0]
        valid_mask=tf.logical_not(tf.math.is_nan(gt_vec[:,:,0:1,:,:]))
        #select pd_vec
        pd_vec_x=pd_vec[:,:,0:1,:,:][valid_mask]
        pd_vec_y=pd_vec[:,:,1:2,:,:][valid_mask]
        pd_vec=tf.stack([pd_vec_x,pd_vec_y])
        #select pd_logb
        pd_logb=pd_logb[:,:,np.newaxis,:,:][valid_mask]
        #select gt_vec
        gt_vec_x=gt_vec[:,:,0:1,:,:][valid_mask]
        gt_vec_y=gt_vec[:,:,1:2,:,:][valid_mask]
        gt_vec=tf.stack([gt_vec_x,gt_vec_y])
        #select gt_bmin
        gt_bmin=gt_bmin[:,:,np.newaxis,:,:][valid_mask]
        #calculate loss
        norm=tf.norm(tf.stack([pd_vec_x-gt_vec_x,pd_vec_y-gt_vec_y,gt_bmin]),axis=0)
        pd_logb=3.0*tf.tanh(pd_logb/3.0)
        scaled_norm=norm*tf.exp(-pd_logb)
        scaled_norm=self.soft_clamp(scaled_norm)
        laplace_loss=pd_logb+scaled_norm
        laplace_loss=tf.reduce_sum(laplace_loss)/batch_size
        return laplace_loss
    
    def Scale_loss(self,pd_scale,gt_scale,b=1.0):
        batch_size=pd_scale.shape[0]
        valid_mask=tf.logical_not(tf.math.is_nan(gt_scale))
        pd_scale=pd_scale[valid_mask]
        pd_scale=tf.nn.softplus(pd_scale)
        gt_scale=gt_scale[valid_mask]
        scale_loss=tf.abs(pd_scale-gt_scale)
        denominator=10.0*(0.1+gt_scale)
        scale_loss=scale_loss/denominator
        scale_loss=self.soft_clamp(scale_loss)
        scale_loss=tf.reduce_sum(scale_loss)/batch_size
        return scale_loss
    
    def cal_loss(self,pd_pif_maps,pd_paf_maps,gt_pif_maps,gt_paf_maps):
        #calculate pif losses
        pd_pif_conf,pd_pif_vec,pd_pif_logb,pd_pif_scale=pd_pif_maps
        gt_pif_conf,gt_pif_vec,gt_pif_bmin,gt_pif_scale=gt_pif_maps
        loss_pif_conf=self.Bce_loss(pd_pif_conf,gt_pif_conf)
        loss_pif_vec=self.Laplace_loss(pd_pif_vec,pd_pif_logb,gt_pif_vec,gt_pif_bmin)
        loss_pif_scale=self.Scale_loss(pd_pif_scale,gt_pif_scale)
        loss_pif_maps=[loss_pif_conf,loss_pif_vec,loss_pif_scale]
        #calculate paf losses
        pd_paf_conf,pd_paf_src_vec,pd_paf_dst_vec,pd_paf_src_logb,pd_paf_dst_logb,pd_paf_src_scale,pd_paf_dst_scale=pd_paf_maps
        gt_paf_conf,gt_paf_src_vec,gt_paf_dst_vec,gt_paf_src_bmin,gt_paf_dst_bmin,gt_paf_src_scale,gt_paf_dst_scale=gt_paf_maps
        loss_paf_conf=self.Bce_loss(pd_paf_conf,gt_paf_conf)
        loss_paf_src_scale=self.Scale_loss(pd_paf_src_scale,gt_paf_src_scale)
        loss_paf_dst_scale=self.Scale_loss(pd_paf_dst_scale,gt_paf_dst_scale)
        loss_paf_src_vec=self.Laplace_loss(pd_paf_src_vec,pd_paf_src_logb,gt_paf_src_vec,gt_paf_src_bmin)
        loss_paf_dst_vec=self.Laplace_loss(pd_paf_dst_vec,pd_paf_dst_logb,gt_paf_dst_vec,gt_paf_dst_bmin)
        loss_paf_maps=[loss_paf_conf,loss_paf_src_vec,loss_paf_dst_vec,loss_paf_src_scale,loss_paf_dst_scale]
        #calculate total loss
        total_loss=(loss_pif_conf*self.lambda_pif_conf+loss_pif_vec*self.lambda_pif_vec+loss_pif_scale*self.lambda_pif_scale+
            loss_paf_conf*self.lambda_paf_conf+loss_paf_src_vec*self.lambda_paf_src_vec+loss_paf_dst_vec*self.lambda_paf_dst_vec+
            loss_paf_src_scale*self.lambda_paf_src_scale+loss_paf_dst_scale*self.lambda_paf_dst_scale)
        #retun losses
        return loss_pif_maps,loss_paf_maps,total_loss
    
    class PifHead(Model):
        def __init__(self,input_features=2048,n_pos=19,n_limbs=19,quad_size=2,hout=8,wout=8,stride=8,mesh_grid=None,data_format="channels_first"):
            super().__init__()
            self.input_features=input_features
            self.n_pos=n_pos
            self.n_limbs=n_limbs
            self.hout=hout
            self.wout=wout
            self.stride=stride
            self.quad_size=quad_size
            self.out_features=self.n_pos*5*(self.quad_size**2)
            self.mesh_grid=mesh_grid
            self.data_format=data_format
            self.tf_data_format="NCHW" if self.data_format=="channels_first" else "NHWC"
            self.main_block=Conv2d(n_filter=self.out_features,in_channels=self.input_features,filter_size=(1,1),data_format=self.data_format)

        def forward(self,x,is_train=False):
            x=self.main_block.forward(x)
            x=tf.nn.depth_to_space(x,block_size=self.quad_size,data_format=self.tf_data_format)
            x=tf.reshape(x,[-1,self.n_pos,5,self.hout,self.wout])
            pif_conf=x[:,:,0,:,:]
            pif_vec=x[:,:,1:3,:,:]
            pif_logb=x[:,:,3,:,:]
            pif_scale=tf.exp(x[:,:,4,:,:])
            #restore vec_maps in inference
            if(is_train==False):
                infer_pif_conf=tf.nn.sigmoid(pif_conf)
                infer_pif_vec=(pif_vec[:,:]+self.mesh_grid)*self.stride
                infer_pif_scale=tf.math.softplus(pif_scale)*self.stride
                return infer_pif_conf,infer_pif_vec,pif_logb,infer_pif_scale
            return pif_conf,pif_vec,pif_logb,pif_scale
        
    class PafHead(Model):
        def __init__(self,input_features=2048,n_pos=19,n_limbs=19,quad_size=2,hout=46,wout=46,stride=8,mesh_grid=None,data_format="channels_first"):
            super().__init__()
            self.input_features=input_features
            self.n_pos=n_pos
            self.n_limbs=n_limbs
            self.quad_size=quad_size
            self.hout=hout
            self.wout=wout
            self.stride=stride
            self.out_features=self.n_limbs*9*(self.quad_size**2)
            self.mesh_grid=mesh_grid
            self.data_format=data_format
            self.tf_data_format="NCHW" if self.data_format=="channels_first" else "NHWC"
            self.main_block=Conv2d(n_filter=self.out_features,in_channels=self.input_features,filter_size=(1,1),data_format=self.data_format)
        
        def forward(self,x,is_train=False):
            x=self.main_block.forward(x)
            x=tf.nn.depth_to_space(x,block_size=self.quad_size,data_format=self.tf_data_format)
            x=tf.reshape(x,[-1,self.n_limbs,9,self.hout,self.wout])
            paf_conf=x[:,:,0,:,:]
            paf_src_vec=x[:,:,1:3,:,:]
            paf_dst_vec=x[:,:,3:5,:,:]
            paf_src_logb=x[:,:,5,:,:]
            paf_dst_logb=x[:,:,6,:,:]
            paf_src_scale=tf.exp(x[:,:,7,:,:])
            paf_dst_scale=tf.exp(x[:,:,8,:,:])
            #restore vec_maps in inference
            if(is_train==False):
                infer_paf_conf=tf.nn.sigmoid(paf_conf)
                infer_paf_src_vec=(paf_src_vec[:,:]+self.mesh_grid)*self.stride
                infer_paf_dst_vec=(paf_dst_vec[:,:]+self.mesh_grid)*self.stride
                infer_paf_src_scale=tf.math.softplus(paf_src_scale)*self.stride
                infer_paf_dst_scale=tf.math.softplus(paf_dst_scale)*self.stride
                return infer_paf_conf,infer_paf_src_vec,infer_paf_dst_vec,paf_src_logb,paf_dst_logb,infer_paf_src_scale,infer_paf_dst_scale
            return paf_conf,paf_src_vec,paf_dst_vec,paf_src_logb,paf_dst_logb,paf_src_scale,paf_dst_scale
