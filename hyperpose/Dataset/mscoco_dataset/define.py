from enum import Enum
import numpy as np
class CocoPart(Enum):
    Nose = 0
    Leye= 1
    Reye= 2
    LEar=3
    REar=4
    LShoulder=5
    RShoulder=6
    LElbow=7
    RElbow=8
    LWrist=9
    RWrist=10
    LHip=11
    RHip=12
    LKnee=13
    RKnee=14
    LAnkle=15
    RAnkle=16

CocoColor = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

#convert kpts from opps to mscoco
from_opps_converter={0:0, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9, 8:12, 9:14, 10:16, 11:11, 12:13, 13:15, 14:2, 15:1, 16:4, 17:3}
#convert kpts from mscoco to opps
to_opps_converter={0:0, 1:15, 2:14, 3:17, 4:16, 5:5, 6:2, 7:6, 8:3, 9:7, 10:4, 11:11, 12:8, 13:12, 14:9, 15:13, 16:10}

def opps_input_converter(coco_kpts):
    cvt_kpts=np.zeros(shape=[19,2])
    transform = np.array(
            list(zip([0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
                     [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3])))
    xs = coco_kpts[0::3]
    ys = coco_kpts[1::3]
    vs = coco_kpts[2::3]
    lost_idx=np.where(vs<=0)[0]
    xs[lost_idx]=-1000
    ys[lost_idx]=-1000
    cvt_xs=(xs[transform[:,0]]+xs[transform[:,1]])/2
    cvt_ys=(ys[transform[:,0]]+ys[transform[:,1]])/2
    cvt_kpts[:-1,:]=np.array([cvt_xs,cvt_ys]).transpose()
    #adding background point
    cvt_kpts[-1:,:]=-1000
    return cvt_kpts

def opps_output_converter(kpt_list):
    kpts=[]
    for coco_idx in list(to_opps_converter.keys()):
        model_idx=to_opps_converter[coco_idx]
        x,y=kpt_list[model_idx]
        if(x<0 or y<0):
            kpts+=[0.0,0.0,0.0]
        else:
            kpts+=[x,y,1.0]
    return kpts

#convert kpts from ppn to mscoco
from_ppn_converter={0:0, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9, 8:12, 9:14, 10:16, 11:11, 12:13, 13:15, 14:2, 15:1, 16:4, 17:3}
#convert kpts from mscoco to ppn
to_ppn_converter={0:0, 1:15, 2:14, 3:17, 4:16, 5:5, 6:2, 7:6, 8:3, 9:7, 10:4, 11:11, 12:8, 13:12, 14:9, 15:13, 16:10}

def ppn_input_converter(coco_kpts):
    transform = np.array(
            list(zip([0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
                     [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3])))
    xs = coco_kpts[0::3]
    ys = coco_kpts[1::3]
    vs = coco_kpts[2::3]
    lost_idx=np.where(vs<=0)[0]
    xs[lost_idx]=-1000
    ys[lost_idx]=-1000
    cvt_xs=(xs[transform[:,0]]+xs[transform[:,1]])/2
    cvt_ys=(ys[transform[:,0]]+ys[transform[:,1]])/2
    cvt_kpts=np.array([cvt_xs,cvt_ys]).transpose()
    return cvt_kpts

def ppn_output_converter(kpt_list):
    kpts=[]
    for coco_idx in list(to_ppn_converter.keys()):
        model_idx=to_ppn_converter[coco_idx]
        x,y=kpt_list[model_idx]
        if(x<0 or y<0):
            kpts+=[0.0,0.0,0.0]
        else:
            kpts+=[x,y,1.0]
    return kpts

#convert kpts from pifpaf to mscoco
from_pifpaf_converter={}
for part_idx in range(0,len(CocoPart)):
    from_pifpaf_converter[part_idx]=part_idx
#convert kpts from mscoco to pifpaf
to_pifpaf_converter={}
for part_idx in range(0,len(CocoPart)):
    to_pifpaf_converter[part_idx]=part_idx

def pifpaf_input_converter(coco_kpts):
    xs=coco_kpts[0::3]
    ys=coco_kpts[1::3]
    vs=coco_kpts[2::3]
    lost_idx=np.where(vs<=0)[0]
    xs[lost_idx]=-1000
    ys[lost_idx]=-1000
    cvt_kpts=np.array([xs,ys]).transpose()
    return cvt_kpts

def pifpaf_output_converter(kpt_list):
    kpts=[]
    for coco_idx in range(0,len(CocoPart)):
        flag=False
        if(coco_idx in to_pifpaf_converter):
            model_idx=to_pifpaf_converter[coco_idx]
            x,y=kpt_list[model_idx]
            if(x>=0 and y>=0):
                kpts+=[x,y,1.0]
                flag=True
        if(not flag):
            kpts+=[0.0,0.0,0.0]
    return kpts