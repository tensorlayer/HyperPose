import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from ..utils import tf_repeat
from ..define import CocoPart,CocoLimb
initializer=tl.initializers.truncated_normal(stddev=0.005)

class LightWeightOpenPose(Model):
    def __init__(self,parts=CocoPart,limbs=CocoLimb,colors=None,n_pos=19,n_limbs=19,num_channels=128,\
        hin=368,win=368,hout=46,wout=46,backbone=None,pretraining=False,data_format="channels_first"):
        super().__init__()
        self.num_channels=num_channels
        self.parts=parts
        self.limbs=limbs
        self.colors=colors
        self.n_pos=n_pos
        self.n_limbs=n_limbs
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
        #dilated mobilenetv1 backbone
        if(backbone==None):
            self.backbone=self.Dilated_mobilenet(data_format=self.data_format)
        else:
            self.backbone=backbone(scale_size=8,pretraining=pretraining,data_format=self.data_format)
        #cpm stage to cutdown dimension
        self.cpm_stage=self.Cpm_stage(n_filter=self.num_channels,in_channels=self.backbone.out_channels,data_format=self.data_format)
        #init stage
        self.init_stage=self.Init_stage(n_filter=self.num_channels,n_confmaps=self.n_confmaps,\
            n_pafmaps=self.n_pafmaps,data_format=self.data_format)
        #one refinemnet stage
        self.refine_stage1=self.Refinement_stage(n_filter=self.num_channels,n_confmaps=self.n_confmaps,n_pafmaps=self.n_pafmaps,\
            in_channels=self.num_channels+self.n_confmaps+self.n_pafmaps,data_format=self.data_format)
    
    @tf.function
    def forward(self,x,is_train=False,stage_num=1,domainadapt=False):
        conf_list=[]
        paf_list=[]
        #backbone feature extract
        backbone_features=self.backbone(x)
        cpm_features=self.cpm_stage(backbone_features)
        #init stage
        init_conf,init_paf=self.init_stage(cpm_features)
        conf_list.append(init_conf)
        paf_list.append(init_paf)
        x=tf.concat([cpm_features,init_conf,init_paf],self.concat_dim)
        #refinement
        ref_conf1,ref_paf1=self.refine_stage1(x)
        conf_list.append(ref_conf1)
        paf_list.append(ref_paf1)
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
        if(self.concat_dim==1):
            mask_conf=tf_repeat(mask, [1,self.n_confmaps ,1,1])
            mask_paf=tf_repeat(mask,[1,self.n_pafmaps ,1,1])
        elif(self.concat_dim==-1):
            mask_conf=tf_repeat(mask, [1,1,1,self.n_confmaps])
            mask_paf=tf_repeat(mask,[1,1,1,self.n_pafmaps])
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

    class Dilated_mobilenet(Model):
        def __init__(self,data_format="channels_first"):
            super().__init__()
            self.data_format=data_format
            self.out_channels=512
            self.scale_size=8
            self.main_block=layers.LayerList([
            conv_block(n_filter=32,in_channels=3,data_format=self.data_format,strides=(2,2)),
            dw_conv_block(n_filter=64,in_channels=32,data_format=self.data_format),
            dw_conv_block(n_filter=128,in_channels=64,data_format=self.data_format,strides=(2,2)),
            dw_conv_block(n_filter=128,in_channels=128,data_format=self.data_format),
            dw_conv_block(n_filter=256,in_channels=128,data_format=self.data_format,strides=(2,2)),
            dw_conv_block(n_filter=256,in_channels=256,data_format=self.data_format),
            dw_conv_block(n_filter=512,in_channels=256,data_format=self.data_format),
            dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format,dilation_rate=(2,2)),
            dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format),
            dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format),
            dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format),
            dw_conv_block(n_filter=512,in_channels=512,data_format=self.data_format)
            ])
        def forward(self,x):
            return self.main_block.forward(x)

    class Cpm_stage(Model):
        def __init__(self,n_filter=128,in_channels=512,data_format="channels_first"):
            super().__init__()
            self.data_format=data_format
            self.init_layer=Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(1,1),act=tf.nn.relu,data_format=self.data_format)
            self.main_block=layers.LayerList([
                conv_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format),
                conv_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format),
                conv_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format),
            ])
            self.end_layer=Conv2d(n_filter=n_filter,in_channels=n_filter,filter_size=(3,3),act=tf.nn.relu,data_format=self.data_format)
        
        def forward(self,x):
            x=self.init_layer.forward(x)
            x=x+self.main_block.forward(x)
            return self.end_layer.forward(x)

    class Init_stage(Model):
        def __init__(self,n_filter=128,n_confmaps=19,n_pafmaps=38,data_format="channels_first"):
            super().__init__()
            self.data_format=data_format
            self.main_block=layers.LayerList([
            Conv2d(n_filter=n_filter,in_channels=n_filter,act=tf.nn.relu,data_format=self.data_format),
            Conv2d(n_filter=n_filter,in_channels=n_filter,act=tf.nn.relu,data_format=self.data_format),
            Conv2d(n_filter=n_filter,in_channels=n_filter,act=tf.nn.relu,data_format=self.data_format)
            ])
            self.conf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,\
                    b_init=initializer,data_format=self.data_format),
            Conv2d(n_filter=n_confmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,\
                    b_init=initializer,data_format=self.data_format)
            ])
            self.paf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,\
                b_init=initializer,data_format=self.data_format),
            Conv2d(n_filter=n_pafmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,\
                b_init=initializer,data_format=self.data_format)
            ])
        def forward(self,x):
            x=self.main_block.forward(x)
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map
    class Refinement_stage(Model):
        def __init__(self,n_filter=128,in_channels=185,n_confmaps=19,n_pafmaps=38,data_format="channels_first"):
            super().__init__()
            self.data_format=data_format
            self.block_1=self.Refinement_block(n_filter=n_filter,in_channels=in_channels,data_format=self.data_format)
            self.block_2=self.Refinement_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format)
            self.block_3=self.Refinement_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format)
            self.block_4=self.Refinement_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format)
            self.block_5=self.Refinement_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format)
            self.conf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,b_init=initializer,\
                data_format=self.data_format),
            Conv2d(n_filter=n_confmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,b_init=initializer,\
                data_format=self.data_format)
            ])
            self.paf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,b_init=initializer,\
                data_format=self.data_format),
            Conv2d(n_filter=n_pafmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,b_init=initializer,\
                data_format=self.data_format)
            ])
        def forward(self,x):
            x=self.block_1(x)
            x=self.block_2(x)
            x=self.block_3(x)
            x=self.block_4(x)
            x=self.block_5(x)
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map        
        class Refinement_block(Model):
            def __init__(self,n_filter,in_channels,data_format="channels_first"):
                super().__init__()
                self.data_format=data_format
                self.init_layer=Conv2d(n_filter=n_filter,filter_size=(1,1),in_channels=in_channels,act=tf.nn.relu,data_format=self.data_format)
                self.main_block=layers.LayerList([
                conv_block(n_filter=n_filter,in_channels=n_filter,data_format=self.data_format),
                conv_block(n_filter=n_filter,in_channels=n_filter,dilation_rate=(1,1),data_format=self.data_format)
                ])
            def forward(self,x):
                x=self.init_layer.forward(x)
                return x+self.main_block.forward(x)

def conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),dilation_rate=(1,1),W_init=initializer,b_init=initializer,padding="SAME",data_format="channels_first"):
    layer_list=[]
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=filter_size,strides=strides,in_channels=in_channels,\
        dilation_rate=dilation_rate,padding=padding,W_init=initializer,b_init=initializer,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99, act=tf.nn.relu,num_features=n_filter,data_format=data_format,is_train=True))
    return layers.LayerList(layer_list)

def dw_conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),dilation_rate=(1,1),W_init=initializer,b_init=initializer,data_format="channels_first"):
    layer_list=[]
    layer_list.append(DepthwiseConv2d(filter_size=filter_size,strides=strides,in_channels=in_channels,
        dilation_rate=dilation_rate,W_init=initializer,b_init=None,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99,act=tf.nn.relu,num_features=in_channels,data_format=data_format,is_train=True))
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=(1,1),strides=(1,1),in_channels=in_channels,W_init=initializer,b_init=None,data_format=data_format))
    layer_list.append(BatchNorm2d(decay=0.99,act=tf.nn.relu,num_features=n_filter,data_format=data_format,is_train=True))
    return layers.LayerList(layer_list)

def nobn_dw_conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),W_init=initializer,b_init=initializer,data_format="channels_first"):
    layer_list=[]
    layer_list.append(DepthwiseConv2d(filter_size=filter_size,strides=strides,in_channels=in_channels,
        act=tf.nn.relu,W_init=initializer,b_init=None,data_format=data_format))
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=(1, 1),strides=(1, 1),in_channels=in_channels,
        act=tf.nn.relu,W_init=initializer,b_init=None,data_format=data_format))
    return layers.LayerList(layer_list)
