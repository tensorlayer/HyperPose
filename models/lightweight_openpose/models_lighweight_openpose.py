import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList
from .utils import tf_repeat
initializer=tl.initializers.truncated_normal(stddev=0.005)

class model(Model):
    def __init__(self,n_pos=19,num_channels=128,hin=368,win=368,hout=46,wout=46):
        super().__init__()
        self.num_channels=num_channels
        self.n_pos=n_pos
        self.num_confmaps=n_pos
        self.num_pafmaps=2*n_pos
        self.hin=hin
        self.win=win
        self.hout=hout
        self.wout=wout
        #dilated mobilenetv1 backbone
        self.backbone=self.Dilated_mobilenet()
        #cpm stage to cutdown dimension
        self.cpm_stage=self.Cpm_stage(n_filter=self.num_channels,in_channels=512)
        #init stage
        self.init_stage=self.Init_stage(n_filter=self.num_channels,num_confmaps=self.num_confmaps,\
            num_pafmaps=self.num_pafmaps)
        #one refinemnet stage
        self.refine_stage1=self.Refinement_stage(n_filter=self.num_channels,\
            in_channels=self.num_channels+self.num_confmaps+self.num_pafmaps)
    
    @tf.function
    def forward(self,x,mask_conf=None,mask_paf=None,is_train=False):
        conf_list=[]
        paf_list=[]
        #backbone feature extract
        backbone_features=self.backbone(x)
        cpm_features=self.cpm_stage(backbone_features)
        #init stage
        init_conf,init_paf=self.init_stage(cpm_features)
        if(is_train):
            init_conf=init_conf*mask_conf
            init_paf=init_paf*mask_paf
        conf_list.append(init_conf)
        paf_list.append(init_paf)
        x=tf.concat([cpm_features,init_conf,init_paf],-1)
        #refinement
        ref_conf1,ref_paf1=self.refine_stage1(x)
        if(is_train):
            ref_conf1=ref_conf1*mask_conf
            ref_paf1=ref_paf1*mask_paf
        conf_list.append(ref_conf1)
        paf_list.append(ref_paf1)
        if(is_train):
            return conf_list[-1],paf_list[-1],conf_list,paf_list
        else:
            return conf_list[-1],paf_list[-1]
    
    def cal_loss(self,gt_conf,gt_paf,mask,stage_confs,stage_pafs):
        stage_losses=[]
        batch_size=gt_conf.shape[0]
        mask_conf=tf_repeat(mask, [1, 1, 1, self.num_confmaps])
        mask_paf=tf_repeat(mask,[1,1,1,self.num_pafmaps])
        for stage_conf,stage_paf in zip(stage_confs,stage_pafs):
            loss_conf=tf.nn.l2_loss((gt_conf-stage_conf)*mask_conf)
            loss_paf=tf.nn.l2_loss((gt_paf-stage_paf)*mask_paf)
            stage_losses.append(loss_conf)
            stage_losses.append(loss_paf)
        pd_loss=tf.reduce_mean(stage_losses)/batch_size
        return pd_loss
    
    class Dilated_mobilenet(Model):
        def __init__(self):
            super().__init__()
            self.main_block=layers.LayerList([
            conv_block(n_filter=32,in_channels=3,strides=(2,2)),
            dw_conv_block(n_filter=64,in_channels=32),
            dw_conv_block(n_filter=128,in_channels=64,strides=(2,2)),
            dw_conv_block(n_filter=128,in_channels=128),
            dw_conv_block(n_filter=256,in_channels=128,strides=(2,2)),
            dw_conv_block(n_filter=256,in_channels=256),
            dw_conv_block(n_filter=512,in_channels=256),
            dw_conv_block(n_filter=512,in_channels=512,dilation_rate=(2,2)),
            dw_conv_block(n_filter=512,in_channels=512),
            dw_conv_block(n_filter=512,in_channels=512),
            dw_conv_block(n_filter=512,in_channels=512),
            dw_conv_block(n_filter=512,in_channels=512)
            ])

        def forward(self,x):
            return self.main_block.forward(x)

    class Cpm_stage(Model):
        def __init__(self,n_filter=128,in_channels=512):
            super().__init__()
            self.init_layer=Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=(1,1),act=tf.nn.relu)
            self.main_block=layers.LayerList([
                nobn_dw_conv_block(n_filter=n_filter,in_channels=n_filter),
                nobn_dw_conv_block(n_filter=n_filter,in_channels=n_filter),
                nobn_dw_conv_block(n_filter=n_filter,in_channels=n_filter),
            ])
            self.end_layer=Conv2d(n_filter=n_filter,in_channels=n_filter,filter_size=(3,3),act=tf.nn.relu)
        
        def forward(self,x):
            x=self.init_layer.forward(x)
            x=x+self.main_block.forward(x)
            return self.end_layer.forward(x)

    class Init_stage(Model):
        def __init__(self,n_filter=128,num_confmaps=19,num_pafmaps=38):
            super().__init__()
            self.main_block=layers.LayerList([
            Conv2d(n_filter=n_filter,in_channels=n_filter,act=tf.nn.relu),
            Conv2d(n_filter=n_filter,in_channels=n_filter,act=tf.nn.relu),
            Conv2d(n_filter=n_filter,in_channels=n_filter,act=tf.nn.relu)
            ])
            self.conf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,b_init=initializer),
            Conv2d(n_filter=num_confmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,b_init=initializer)
            ])
            self.paf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,b_init=initializer),
            Conv2d(n_filter=num_pafmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,b_init=initializer)
            ])

        def forward(self,x):
            x=self.main_block.forward(x)
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map
   
    class Refinement_stage(Model):
        def __init__(self,n_filter=128,in_channels=185,num_confmaps=19,num_pafmaps=38):
            super().__init__()
            self.block_1=self.Refinement_block(n_filter=n_filter,in_channels=in_channels)
            self.block_2=self.Refinement_block(n_filter=n_filter,in_channels=n_filter)
            self.block_3=self.Refinement_block(n_filter=n_filter,in_channels=n_filter)
            self.block_4=self.Refinement_block(n_filter=n_filter,in_channels=n_filter)
            self.block_5=self.Refinement_block(n_filter=n_filter,in_channels=n_filter)

            self.conf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,b_init=initializer),
            Conv2d(n_filter=num_confmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,b_init=initializer)
            ])
            self.paf_block=layers.LayerList([
            Conv2d(n_filter=512,in_channels=n_filter,filter_size=(1,1),strides=(1,1),act=tf.nn.relu,W_init=initializer,b_init=initializer),
            Conv2d(n_filter=num_pafmaps,in_channels=512,filter_size=(1,1),strides=(1,1),W_init=initializer,b_init=initializer)
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
            def __init__(self,n_filter,in_channels):
                super().__init__()
                self.init_layer=Conv2d(n_filter=n_filter,filter_size=(1,1),in_channels=in_channels,act=tf.nn.relu)
                self.main_block=layers.LayerList([
                conv_block(n_filter=n_filter,in_channels=n_filter),
                conv_block(n_filter=n_filter,in_channels=n_filter,dilation_rate=(2,2))
                ])
                
            def forward(self,x):
                x=self.init_layer.forward(x)
                return x+self.main_block.forward(x)

def conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),dilation_rate=(1,1),W_init=initializer,b_init=initializer,padding="SAME"):
    layer_list=[]
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=filter_size,strides=strides,in_channels=in_channels,\
        dilation_rate=dilation_rate,padding=padding,W_init=initializer,b_init=initializer))
    layer_list.append(BatchNorm2d(decay=0.99, act=tf.nn.relu,num_features=n_filter))
    return layers.LayerList(layer_list)

def dw_conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),dilation_rate=(1,1),W_init=initializer,b_init=initializer):
    layer_list=[]
    layer_list.append(DepthwiseConv2d(filter_size=filter_size,strides=strides,in_channels=in_channels,
        dilation_rate=dilation_rate,W_init=initializer,b_init=None))
    layer_list.append(BatchNorm2d(decay=0.99,act=tf.nn.relu,num_features=in_channels))
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=(1,1),strides=(1,1),in_channels=in_channels,W_init=initializer,b_init=None))
    layer_list.append(BatchNorm2d(decay=0.99,act=tf.nn.relu,num_features=n_filter))
    return layers.LayerList(layer_list)

def nobn_dw_conv_block(n_filter,in_channels,filter_size=(3,3),strides=(1,1),W_init=initializer,b_init=initializer):
    layer_list=[]
    layer_list.append(DepthwiseConv2d(filter_size=filter_size,strides=strides,in_channels=in_channels,
        act=tf.nn.relu,W_init=initializer,b_init=None))
    layer_list.append(Conv2d(n_filter=n_filter,filter_size=(1, 1),strides=(1, 1),in_channels=in_channels,
        act=tf.nn.relu,W_init=initializer,b_init=None))
    return layers.LayerList(layer_list)
