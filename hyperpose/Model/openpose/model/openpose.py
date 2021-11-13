import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from ..utils import NCHW_to_NHWC, NHWC_to_NCHW
from ..define import CocoPart,CocoLimb
from ...backbones import vgg19_backbone
from ...common import regulize_loss
initial_w=tl.initializers.random_normal(stddev=0.001)
initial_b=tl.initializers.constant(value=0.0)

class OpenPose(Model):
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
        self.concat_dim=1 if self.data_format=="channels_first" else -1
        #back bone configure
        if(backbone==None):
            self.backbone=vgg19_backbone(scale_size=8,pretraining=pretraining,data_format=self.data_format)
        else:
            self.backbone=backbone(scale_size=8,pretraining=pretraining,data_format=self.data_format)
        self.cpm_stage=LayerList([
            Conv2d(n_filter=256,in_channels=self.backbone.out_channels,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,data_format=self.data_format),
            Conv2d(n_filter=128,in_channels=256,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,data_format=self.data_format)
        ])
        #init stage
        self.init_stage=self.Init_stage(n_confmaps=self.n_confmaps, n_pafmaps=self.n_pafmaps,in_channels=128,data_format=self.data_format)
        #one refinemnet stage
        self.refinement_stage_1=self.Refinement_stage(n_confmaps=self.n_confmaps, n_pafmaps=self.n_pafmaps, in_channels=self.n_confmaps+self.n_pafmaps+128,data_format=self.data_format)
        self.refinement_stage_2=self.Refinement_stage(n_confmaps=self.n_confmaps, n_pafmaps=self.n_pafmaps, in_channels=self.n_confmaps+self.n_pafmaps+128,data_format=self.data_format)
        self.refinement_stage_3=self.Refinement_stage(n_confmaps=self.n_confmaps, n_pafmaps=self.n_pafmaps, in_channels=self.n_confmaps+self.n_pafmaps+128,data_format=self.data_format)
        self.refinement_stage_4=self.Refinement_stage(n_confmaps=self.n_confmaps, n_pafmaps=self.n_pafmaps, in_channels=self.n_confmaps+self.n_pafmaps+128,data_format=self.data_format)
        self.refinement_stage_5=self.Refinement_stage(n_confmaps=self.n_confmaps, n_pafmaps=self.n_pafmaps, in_channels=self.n_confmaps+self.n_pafmaps+128,data_format=self.data_format)
        

    @tf.function
    def forward(self,x,is_train=False,ret_backbone=False):
        if(self.data_format == "channels_last"):
            x = NCHW_to_NHWC(x)
        stage_num=5
        conf_list=[]
        paf_list=[]
        # backbone feature extract
        backbone_features=self.backbone.forward(x)
        backbone_features=self.cpm_stage.forward(backbone_features)
        # init stage
        init_conf,init_paf=self.init_stage.forward(backbone_features)
        conf_list.append(init_conf)
        paf_list.append(init_paf)
        # refinement stages  
        for refine_stage_idx in range(1,stage_num+1):
            ref_x=tf.concat([backbone_features,conf_list[-1],paf_list[-1]],self.concat_dim)
            ref_conf,ref_paf=eval(f"self.refinement_stage_{refine_stage_idx}.forward(ref_x)")
            conf_list.append(ref_conf)
            paf_list.append(ref_paf)
        
        if(self.data_format == "channels_last"):
            backbone_features = NHWC_to_NCHW(backbone_features)
            conf_list = [NHWC_to_NCHW(conf) for conf in conf_list]
            paf_list = [NHWC_to_NCHW(paf) for paf in paf_list]
        
        # construct predict_x
        predict_x = {"conf_map": conf_list[-1], "paf_map": paf_list[-1], "stage_confs": conf_list, "stage_pafs": paf_list}
        if(ret_backbone):
            predict_x["backbone_features"]=backbone_features

        return predict_x
    
    @tf.function(experimental_relax_shapes=True)
    def infer(self,x):
        predict_x = self.forward(x,is_train=False)
        conf_map, paf_map = predict_x["conf_map"], predict_x["paf_map"]
        return conf_map, paf_map
    
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
        for stage_id,(stage_conf,stage_paf) in enumerate(zip(stage_confs,stage_pafs)):
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
        def __init__(self,n_confmaps=19,n_pafmaps=38,in_channels=128,data_format="channels_first"):
            super().__init__()
            self.n_confmaps=n_confmaps
            self.n_pafmaps=n_pafmaps
            self.in_channels=in_channels
            self.data_format=data_format
            self.conf_block=layers.LayerList([
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=512,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=512),
                Conv2d(n_filter=self.n_confmaps,in_channels=512,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=self.n_confmaps)
            ])
            self.paf_block=layers.LayerList([
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=512,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=512),
                Conv2d(n_filter=self.n_pafmaps,in_channels=512,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=self.n_pafmaps)
            ])
        
        def forward(self,x):
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map
    
    class Refinement_stage(Model):
        def __init__(self,n_confmaps=19,n_pafmaps=38,in_channels=185,data_format="channels_first"):
            super().__init__()
            self.n_confmaps=n_confmaps
            self.n_pafmaps=n_pafmaps
            self.in_channels=in_channels
            self.data_format=data_format
            self.conf_block=layers.LayerList([
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=self.n_confmaps,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=self.n_confmaps)
            ])
            self.paf_block=layers.LayerList([
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=128,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=128),
                Conv2d(n_filter=self.n_pafmaps,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=None,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                tl.layers.PRelu(in_channels=self.n_pafmaps)
            ])
        
        def forward(self,x):
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map