import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from ..backbones import Resnet50_backbone

class Pifpaf(Model):
    def __init__(self,parts,limbs,n_pos=17,n_limbs=19,hin=368,win=368,scale_size=8,backbone=None,pretraining=False,quad_size=2,
    lambda_pif_conf=30.0,lambda_pif_vec=2.0,lambda_pif_scale=2.0,lambda_paf_conf=50.0,lambda_paf_src_vec=3.0,lambda_paf_dst_vec=3.0,data_format="channels_first"):
        super().__init__()
        self.parts=parts
        self.limbs=limbs
        self.n_pos=n_pos
        self.n_limbs=n_limbs
        self.hin=hin
        self.win=win
        self.hout=int(hin/scale_size)
        self.wout=int(win/scale_size)
        self.quad_size=quad_size
        self.stride=scale_size
        self.scale_size=scale_size*(2**self.quad_size)
        self.lambda_pif_conf=lambda_pif_conf
        self.lambda_pif_vec=lambda_pif_vec
        self.lambda_pif_scale=lambda_pif_scale
        self.lambda_paf_conf=lambda_paf_conf
        self.lambda_paf_src_vec=lambda_paf_src_vec
        self.lambda_paf_dst_vec=lambda_paf_dst_vec
        self.data_format=data_format
        if(backbone==None):
            self.backbone=Resnet50_backbone(data_format=data_format,use_pool=False,scale_size=32)
        else:
            self.backbone=backbone(data_format=data_format,scale_size=self.scale_size)
        self.pif_head=self.PifHead(input_features=self.backbone.out_channels,n_pos=self.n_pos,n_limbs=self.n_limbs,\
            quad_size=self.quad_size,data_format=self.data_format)
        self.paf_head=self.PafHead(input_features=self.backbone.out_channels,n_pos=self.n_pos,n_limbs=self.n_limbs,\
            quad_size=self.quad_size,data_format=self.data_format)
        #generate mesh grid
        x_range=np.linspace(start=0,stop=self.wout-1,num=self.wout)
        y_range=np.linspace(start=0,stop=self.hout-1,num=self.hout)
        mesh_x,mesh_y=np.meshgrid(x_range,y_range)
        self.mesh_grid=np.stack([mesh_x,mesh_y])
    
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
        bce_loss=tf.clip_by_value(bce_loss,0.02,5.0)
        if(focal_gamma!=0.0):
            focal=(1-tf.exp(-bce_loss))**focal_gamma
            focal=tf.stop_gradient(focal)
            bce_loss=focal*bce_loss
        bce_loss=tf.reduce_sum(bce_loss)/batch_size
        return bce_loss
    
    def Laplace_loss(self,pd_vec,pd_logb,gt_vec):
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
        #calculate loss
        norm=tf.norm(pd_vec-gt_vec,axis=0)
        norm=tf.clip_by_value(norm,0.0,5.0)
        pd_logb=tf.clip_by_value(pd_logb,-3.0,np.inf)
        laplace_loss=pd_logb+(norm+0.1)*tf.exp(-pd_logb)
        laplace_loss=tf.reduce_sum(laplace_loss)/batch_size
        return laplace_loss
    
    def Scale_loss(self,pd_scale,gt_scale,b=1.0):
        batch_size=pd_scale.shape[0]
        valid_mask=tf.logical_not(tf.math.is_nan(gt_scale))
        pd_scale=pd_scale[valid_mask]
        gt_scale=gt_scale[valid_mask]
        scale_loss=tf.abs(pd_scale-gt_scale)
        scale_loss=tf.clip_by_value(scale_loss,0.0,5.0)/b
        scale_loss=tf.reduce_sum(scale_loss)/batch_size
        return scale_loss
    
    def cal_loss(self,pd_pif_maps,pd_paf_maps,gt_pif_maps,gt_paf_maps):
        #calculate pif losses
        pd_pif_conf,pd_pif_vec,pd_pif_logb,pd_pif_scale=pd_pif_maps
        gt_pif_conf,gt_pif_vec,gt_pif_scale=gt_pif_maps
        loss_pif_conf=self.Bce_loss(pd_pif_conf,gt_pif_conf)
        loss_pif_vec=self.Laplace_loss(pd_pif_vec,pd_pif_logb,gt_pif_vec)
        loss_pif_scale=self.Scale_loss(pd_pif_scale,gt_pif_scale)
        loss_pif_maps=[loss_pif_conf,loss_pif_vec,loss_pif_scale]
        #calculate paf losses
        pd_paf_conf,pd_paf_src_vec,pd_paf_dst_vec,pd_paf_src_logb,pd_paf_dst_logb,pd_paf_src_scale,pd_paf_dst_scale=pd_paf_maps
        gt_paf_conf,gt_paf_src_vec,gt_paf_dst_vec,gt_paf_src_scale,gt_paf_dst_scale=gt_paf_maps
        loss_paf_conf=self.Bce_loss(pd_paf_conf,gt_paf_conf)
        loss_paf_src_vec=self.Laplace_loss(pd_paf_src_vec,pd_paf_src_logb,gt_paf_src_vec)
        loss_paf_dst_vec=self.Laplace_loss(pd_paf_dst_vec,pd_paf_dst_logb,gt_paf_dst_vec)
        loss_paf_maps=[loss_paf_conf,loss_paf_src_vec,loss_paf_dst_vec]
        #calculate total loss
        total_loss=(loss_pif_conf*self.lambda_pif_conf+loss_pif_vec*self.lambda_pif_vec+loss_pif_scale*self.lambda_pif_scale+
            loss_paf_conf*self.lambda_paf_conf+loss_paf_src_vec*self.lambda_paf_src_vec+loss_paf_dst_vec*self.lambda_paf_dst_vec)
        #retun losses
        return loss_pif_maps,loss_paf_maps,total_loss
    
    class PifHead(Model):
        def __init__(self,input_features=2048,n_pos=19,n_limbs=19,quad_size=2,data_format="channels_first"):
            super().__init__()
            self.input_features=input_features
            self.n_pos=n_pos
            self.n_limbs=n_limbs
            self.quad_size=quad_size
            self.out_features=self.n_pos*5*(2**self.quad_size)
            self.data_format=data_format
            self.tf_data_format="NCHW" if self.data_format=="channels_first" else "NHWC"
            self.main_block=Conv2d(n_filter=self.out_features,in_channels=self.input_features,filter_size=(1,1),data_format=self.data_format)

        def forward(self,x,is_train=False):
            x=self.main_block.forward(x)
            x=tf.nn.depth_to_space(x,block_size=self.quad_size,data_format=self.tf_data_format)
            x=tf.reshape(x,[x.shape[0],self.n_pos,5,x.shape[2],x.shape[3]])
            pif_conf=x[:,:,0,:,:]
            pif_vec=x[:,:,1:3,:,:]
            pif_logb=x[:,:,3,:,:]
            pif_scale=tf.exp(x[:,:,4,:,:])
            #restore vec_maps in inference
            if(is_train==False):
                pif_conf=tf.nn.sigmoid(pif_conf)
                pif_vec[:,:]+=self.mesh_grid
            return pif_conf,pif_vec,pif_logb,pif_scale
        
    class PafHead(Model):
        def __init__(self,input_features=2048,n_pos=19,n_limbs=19,quad_size=2,data_format="channels_first"):
            super().__init__()
            self.input_features=input_features
            self.n_pos=n_pos
            self.n_limbs=n_limbs
            self.quad_size=quad_size
            self.out_features=self.n_limbs*9*(2**self.quad_size)
            self.data_format=data_format
            self.tf_data_format="NCHW" if self.data_format=="channels_first" else "NHWC"
            self.main_block=Conv2d(n_filter=self.out_features,in_channels=self.input_features,filter_size=(1,1),data_format=self.data_format)
        
        def forward(self,x,is_train=False):
            x=self.main_block.forward(x)
            x=tf.nn.depth_to_space(x,block_size=self.quad_size,data_format=self.tf_data_format)
            x=tf.reshape(x,[x.shape[0],self.n_limbs,9,x.shape[2],x.shape[3]])
            paf_conf=x[:,:,0,:,:]
            paf_src_vec=x[:,:,1:3,:,:]
            paf_dst_vec=x[:,:,3:5,:,:]
            paf_src_logb=x[:,:,5,:,:]
            paf_dst_logb=x[:,:,6,:,:]
            paf_src_scale=tf.exp(x[:,:,7,:,:])
            paf_dst_scale=tf.exp(x[:,:,8,:,:])
            #restore vec_maps in inference
            if(is_train==False):
                paf_conf=tf.nn.sigmoid(paf_conf)
                paf_src_vec[:,:]+=self.mesh_grid
                paf_dst_vec[:,:]+=self.mesh_grid
            return paf_conf,paf_src_vec,paf_dst_vec,paf_src_logb,paf_dst_logb,paf_src_scale,paf_dst_scale
