import tensorflow as tf
import tensorlayer as tl
from tensorlayer import layers
from tensorlayer.models import Model
from tensorlayer.layers import BatchNorm2d, Conv2d, DepthwiseConv2d, LayerList, MaxPool2d
from ..utils import tf_repeat
from ..define import CocoPart,CocoLimb
initial_w=tl.initializers.random_normal(stddev=0.01)
initial_b=tl.initializers.constant(value=0.0)

class OpenPose(Model):
    def __init__(self,parts=CocoPart,limbs=CocoLimb,colors=None,n_pos=19,n_limbs=19,num_channels=128,hin=368,win=368,hout=46,wout=46,backbone=None,pretrained_backbone=True,data_format="channels_first"):
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
        self.pretrained_backbone=pretrained_backbone
        self.concat_dim=1 if self.data_format=="channels_first" else -1
        #back bone configure
        if(backbone==None):
            self.backbone=self.vgg19(in_channels=3,pretrained=self.pretrained_backbone,data_format=self.data_format)
        else:
            self.backbone=backbone(scale_size=8,data_format=self.data_format)
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
    def forward(self,x,is_train=False):
        conf_list=[]
        paf_list=[]
        #backbone feature extract
        vgg_features=self.backbone.forward(x)
        vgg_features=self.cpm_stage.forward(vgg_features)
        #init stage
        init_conf,init_paf=self.init_stage.forward(vgg_features)
        conf_list.append(init_conf)
        paf_list.append(init_paf)
        #refinement stages
        for refine_stage_idx in range(1,6):
            ref_x=tf.concat([vgg_features,conf_list[-1],paf_list[-1]],self.concat_dim)
            ref_conf,ref_paf=eval(f"self.refinement_stage_{refine_stage_idx}.forward(ref_x)")
            conf_list.append(ref_conf)
            paf_list.append(ref_paf)
        if(is_train):
            return conf_list[-1],paf_list[-1],conf_list,paf_list
        else:
            return conf_list[-1],paf_list[-1]
    
    @tf.function
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
        for stage_id,(stage_conf,stage_paf) in enumerate(zip(stage_confs,stage_pafs)):
            loss_conf=tf.nn.l2_loss((gt_conf-stage_conf)*mask_conf)
            loss_paf=tf.nn.l2_loss((gt_paf-stage_paf)*mask_paf)
            stage_losses.append(loss_conf)
            stage_losses.append(loss_paf)
            loss_confs.append(loss_conf)
            loss_pafs.append(loss_paf)
        pd_loss=tf.reduce_mean(stage_losses)/batch_size
        return pd_loss,loss_confs,loss_pafs
     
    class vgg19(Model):
        def __init__(self,in_channels=3,data_format="channels_first",pretrained=True):
            super().__init__()
            self.data_format=data_format
            self.pretrained=pretrained
            self.transpose=False
            self.out_channels=512
            if(self.data_format=="channel_last"):
                self.main_block=tl.models.vgg19(pretrained=self.pretrained,end_with="conv4_2")
            else:
                if(self.pretrained):
                    print("only channels_last pretrained vgg19 available, adding transpose")
                    self.main_block=tl.models.vgg19(pretrained=self.pretrained,end_with="conv4_2")
                    self.transpose=True
                else:
                    self.main_block=layers.LayerList([
                    self.conv_block(n_filter=64,in_channels=3,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    self.conv_block(n_filter=64,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
                    self.conv_block(n_filter=128,in_channels=64,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    self.conv_block(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
                    self.conv_block(n_filter=256,in_channels=128,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    self.conv_block(n_filter=256,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    MaxPool2d(filter_size=(2,2),strides=(2,2),data_format=self.data_format),
                    self.conv_block(n_filter=512,in_channels=256,filter_size=(3,3),strides=(1,1),act=tf.nn.relu),
                    self.conv_block(n_filter=512,in_channels=512,filter_size=(3,3),strides=(1,1),act=tf.nn.relu)
                    ])

        def conv_block(self,n_filter=32,in_channels=3,filter_size=(3,3),strides=(1,1),act=None,padding="SAME"):
            return Conv2d(n_filter=n_filter,in_channels=in_channels,filter_size=filter_size,strides=strides,\
                act=act,data_format=self.data_format,padding=padding)

        def forward(self,x):
            if(self.transpose):
                x=tf.transpose(x,[0,2,3,1])
            x=self.main_block.forward(x)
            if(self.transpose):
                x=tf.transpose(x,[0,3,1,2])
            return x
    
    class Init_stage(Model):
        def __init__(self,n_confmaps=19,n_pafmaps=38,in_channels=128,data_format="channels_first"):
            super().__init__()
            self.n_confmaps=n_confmaps
            self.n_pafmaps=n_pafmaps
            self.in_channels=in_channels
            self.data_format=data_format
            self.conf_block=layers.LayerList([
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=512,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=self.n_confmaps,in_channels=512,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format)
            ])
            self.paf_block=layers.LayerList([
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(3,3),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=512,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=self.n_pafmaps,in_channels=512,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format)
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
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=self.n_confmaps,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format)
            ])
            self.paf_block=layers.LayerList([
                Conv2d(n_filter=128,in_channels=self.in_channels,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(7,7),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=128,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format),
                Conv2d(n_filter=self.n_pafmaps,in_channels=128,filter_size=(1,1),strides=(1,1),padding="SAME",act=tf.nn.relu,W_init=initial_w,b_init=initial_b,data_format=self.data_format)
            ])
        
        def forward(self,x):
            conf_map=self.conf_block.forward(x)
            paf_map=self.paf_block.forward(x)
            return conf_map,paf_map