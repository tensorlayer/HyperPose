import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d ,SeparableConv2d, UpSampling2d
from ..utils import tf_repeat
from ..define import CocoPart,CocoLimb

class Mobilenetv2_small_Openpose(Model):
    def __init__(self,parts=CocoPart,limbs=CocoLimb,colors=None,n_pos=19,n_limbs=19,num_channels=128,\
        hin=368,win=368,hout=46,wout=46,backbone=None,pretraining=False,data_format="channels_first"):
        super().__init__()
        self.num_channels=num_channels
        self.parts=parts
        self.limbs=limbs
        self.n_pos=n_pos
        self.n_limbs=n_limbs
        self.colors=colors
        self.n_confmaps=n_pos
        self.n_pafmaps=2*n_limbs
        self.hin=hin
        self.win=win
        self.hout=hout
        self.wout=wout
        self.data_format=data_format
        if(self.data_format=="channels_first"):
            self.concat_dim=1
        else:
            self.concat_dim=-1
        if(backbone==None):
            self.backbone=self.Mobilenetv2_variant(data_format=self.data_format)
        else:
            self.backbone=backbone(scale_size=8,pretraining=pretraining,data_format=self.data_format)
        self.init_stage=self.Init_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels,data_format=self.data_format)
        self.refinement_stage_1=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
        self.refinement_stage_2=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
        self.refinement_stage_3=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
        self.refinement_stage_4=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
    
    @tf.function
    def forward(self,x,mask_conf=None,mask_paf=None,is_train=False,stage_num=4,domainadapt=False):
        conf_list=[]
        paf_list=[] 
        backbone_features=self.backbone.forward(x)
        conf_map,paf_map=self.init_stage.forward(backbone_features)
        conf_list.append(conf_map)
        paf_list.append(paf_map)
        for refinement_stage_idx in range(1,stage_num+1):
            x=tf.concat([backbone_features,conf_list[-1],paf_list[-1]],self.concat_dim)
            conf_map,paf_map=eval(f"self.refinement_stage_{refinement_stage_idx}.forward(x)")
            conf_list.append(conf_map)
            paf_list.append(paf_map)
        if(domainadapt):
            return conf_list[-1],paf_list[-1],conf_list,paf_list,backbone_features
        elif(is_train):
            return conf_list[-1],paf_list[-1],conf_list,paf_list
        else:
            return conf_list[-1],paf_list[-1]
    
    @tf.function(experimental_relax_shapes=True)
    def infer(self,x):
        conf_map,paf_map=self.forward(x,is_train=False)
        return conf_map,paf_map
    
    def cal_loss(self,gt_conf,gt_paf,mask,stage_confs,stage_pafs):
        stage_losses=[]
        batch_size=gt_conf.shape[0]
        mask_conf=tf_repeat(mask, [1,self.n_confmaps ,1,1])
        mask_paf=tf_repeat(mask,[1,self.n_pafmaps ,1,1])
        loss_confs,loss_pafs=[],[]
        for stage_conf,stage_paf in zip(stage_confs,stage_pafs):
            loss_conf=tf.nn.l2_loss((gt_conf-stage_conf)*mask_conf)
            loss_paf=tf.nn.l2_loss((gt_paf-stage_paf)*mask_paf)
            stage_losses.append(loss_conf)
            stage_losses.append(loss_paf)
            loss_confs.append(loss_conf)
            loss_pafs.append(loss_paf)
        pd_loss=tf.reduce_mean(stage_losses)/batch_size
        return pd_loss,loss_confs,loss_pafs

    class Mobilenetv2_variant(Model):
        def __init__(self,data_format="channels_first"):
            super().__init__()
            self.data_format=data_format
            if(self.data_format=="channels_first"):
                self.concat_dim=1
            else:
                self.concat_dim=-1
            self.out_channels=704
            self.scale_size=8
            self.convblock_0=conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
            self.convblock_1=separable_block(n_filter=64,in_channels=32,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
            self.convblock_2=separable_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
            self.convblock_3=separable_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
            self.convblock_4=separable_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(2,2),act=tf.nn.relu,data_format=self.data_format)
            self.convblock_5=separable_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
            self.convblock_6=separable_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
            self.convblock_7=separable_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format)
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
        
    class Init_stage(Model):
        def __init__(self,n_confmaps=19,n_pafmaps=38,in_channels=704,data_format="channels_first"):
            self.n_confmaps=n_confmaps
            self.n_pafmaps=n_pafmaps
            self.in_channels=in_channels
            self.data_format=data_format
            #conf block
            self.conf_block=LayerList([
                separable_block(n_filter=128,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=512,in_channels=128,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=self.n_confmaps,in_channels=512,filter_size=(1,1),strides=(1,1),act=None,data_format=self.data_format)
            ])
            #paf block
            self.paf_block=LayerList([
                separable_block(n_filter=128,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=512,in_channels=128,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=self.n_pafmaps,in_channels=512,filter_size=(1,1),strides=(1,1),act=None,data_format=self.data_format)
            ])
        
        def forward(self,x):
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map
        
    class Refinement_stage(Model):
        def __init__(self,n_confmaps=19,n_pafmaps=38,in_channels=19+38+704,data_format="channels_first"):
            self.n_confmaps=n_confmaps
            self.n_pafmaps=n_pafmaps
            self.in_channels=in_channels
            self.data_format=data_format
            #conf_block
            self.conf_block=LayerList([
                separable_block(n_filter=128,in_channels=self.in_channels,filter_size=(7,7),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=self.n_confmaps,in_channels=128,filter_size=(1,1),strides=(1,1),act=None,data_format=self.data_format),
            ])
            #paf_block
            self.conf_block=LayerList([
                separable_block(n_filter=128,in_channels=self.in_channels,filter_size=(7,7),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=128,in_channels=128,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,data_format=self.data_format),
                separable_block(n_filter=self.n_pafmaps,in_channels=128,filter_size=(1,1),strides=(1,1),act=None,data_format=self.data_format),
            ])
        
        def forward(self,x):
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map

def conv_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,padding="SAME",data_format="channels_first"):
    layer_list=[]
    layer_list.append(Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,act=act,\
        data_format=data_format,padding=padding))
    layer_list.append(BatchNorm2d(num_features=n_filter,decay=0.999,is_train=True,act=act,data_format=data_format))
    return LayerList(layer_list)

def separable_block(n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=tf.nn.relu,padding="SAME",data_format="channels_first"):
    layer_list=[]
    layer_list.append(SeparableConv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,act=act,\
        data_format=data_format,padding=padding))
    layer_list.append(BatchNorm2d(num_features=n_filter,decay=0.999,is_train=True,act=act,data_format=data_format))
    return LayerList(layer_list)