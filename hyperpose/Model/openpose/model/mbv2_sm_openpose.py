import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, LayerList ,SeparableConv2d
from ..utils import tf_repeat
from ..define import CocoPart,CocoLimb
from ...common import regulize_loss
from ...backbones import MobilenetSmall_backbone

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
            self.backbone=MobilenetSmall_backbone(data_format=self.data_format)
        else:
            self.backbone=backbone(scale_size=8,pretraining=pretraining,data_format=self.data_format)
        self.init_stage=self.Init_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels,data_format=self.data_format)
        self.refinement_stage_1=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
        self.refinement_stage_2=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
        self.refinement_stage_3=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
        self.refinement_stage_4=self.Refinement_stage(n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,in_channels=self.backbone.out_channels+3*self.n_confmaps,data_format=self.data_format)
    
    @tf.function
    def forward(self, x, is_train=False, ret_backbone=False):
        stage_num=4
        conf_list=[]
        paf_list=[] 
        # backbone feature extract
        backbone_features=self.backbone.forward(x)
        # init stage
        conf_map,paf_map=self.init_stage.forward(backbone_features)
        conf_list.append(conf_map)
        paf_list.append(paf_map)
        # refinement
        for refinement_stage_idx in range(1,stage_num+1):
            x=tf.concat([backbone_features,conf_list[-1],paf_list[-1]],self.concat_dim)
            conf_map,paf_map=eval(f"self.refinement_stage_{refinement_stage_idx}.forward(x)")
            conf_list.append(conf_map)
            paf_list.append(paf_map)
        
        # construct predict_x
        predict_x = {"conf_map": conf_list[-1], "paf_map": paf_list[-1], "stage_confs": conf_list, "stage_pafs": paf_list}
        if(ret_backbone):
            predict_x["backbone_features"]=backbone_features
        
        return predict_x
    
    @tf.function(experimental_relax_shapes=True)
    def infer(self,x):
        predict_x = self.forward(x,is_train=False)
        conf_map, paf_map = predict_x["conf_map"],predict_x["paf_map"]
        return conf_map,paf_map
    
    def cal_loss(self, predict_x, target_x, metric_manager, mask=None):
        # TODO: exclude the loss calculate from mask
        # predict maps
        stage_confs = predict_x["stage_confs"]
        stage_pafs = predict_x["stage_pafs"]
        # target maps
        gt_conf = target_x["conf_map"]
        gt_paf = target_x["paf_map"]

        stage_losses=[]
        batch_size=gt_conf.shape[0]
        loss_confs,loss_pafs=[],[]
        for stage_conf,stage_paf in zip(stage_confs,stage_pafs):
            loss_conf=tf.nn.l2_loss(gt_conf-stage_conf)
            loss_paf=tf.nn.l2_loss(gt_paf-stage_paf)
            stage_losses.append(loss_conf)
            stage_losses.append(loss_paf)
            loss_confs.append(loss_conf)
            loss_pafs.append(loss_paf)
        pd_loss=tf.reduce_mean(stage_losses)/batch_size
        total_loss = pd_loss
        metric_manager.update("model/conf_loss",loss_confs[-1])
        metric_manager.update("model/paf_loss",loss_pafs[-1])
        # regularize loss
        regularize_loss = regulize_loss(self, weight_decay_factor=2e-4)
        total_loss += regularize_loss
        metric_manager.update("model/loss_re",regularize_loss)

        return total_loss

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