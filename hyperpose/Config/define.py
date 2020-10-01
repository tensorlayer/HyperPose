from enum import Enum

class BACKBONE(Enum):
    Default=0
    Mobilenetv1=1
    Vgg19=2
    Resnet18=3
    Resnet50=4
    Vggtiny=5
    Mobilenetv2=6
    Vgg16=7

class MODEL(Enum):
    Openpose=0
    LightweightOpenpose=1
    PoseProposal=2
    MobilenetThinOpenpose=3
    Pifpaf=4

class DATA(Enum):
    MSCOCO=0
    MPII=1
    USERDEF=2
    MULTIPLE=3

class TRAIN(Enum):
    Single_train=0
    Parallel_train=1

class KUNGFU(Enum):
    Sync_sgd=0
    Sync_avg=1
    Pair_avg=2

class OPTIM(Enum):
    Adam=0
    RMSprop=1
    SGD=2
