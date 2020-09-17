from enum import Enum
import numpy as np

class MpiiPart(Enum):
    RAnkle = 0
    RKnee = 1
    RHip = 2
    LHip = 3
    LKnee = 4
    LAnkle = 5
    Pelvis=6
    Thorax=7
    UpperNeck=8
    Headtop=9
    RWrist =10
    RElbow =11
    RShoulder =12
    LShoulder =13
    LElbow = 14
    LWrist = 15

    @staticmethod
    def from_coco(human):
        from ..mscoco_dataset.define import CocoPart
        t = [
            (MpiiPart.Head, CocoPart.Nose),
            (MpiiPart.Neck, CocoPart.Neck),
            (MpiiPart.RShoulder, CocoPart.RShoulder),
            (MpiiPart.RElbow, CocoPart.RElbow),
            (MpiiPart.RWrist, CocoPart.RWrist),
            (MpiiPart.LShoulder, CocoPart.LShoulder),
            (MpiiPart.LElbow, CocoPart.LElbow),
            (MpiiPart.LWrist, CocoPart.LWrist),
            (MpiiPart.RHip, CocoPart.RHip),
            (MpiiPart.RKnee, CocoPart.RKnee),
            (MpiiPart.RAnkle, CocoPart.RAnkle),
            (MpiiPart.LHip, CocoPart.LHip),
            (MpiiPart.LKnee, CocoPart.LKnee),
            (MpiiPart.LAnkle, CocoPart.LAnkle),
        ]

        pose_2d_mpii = []
        visibilty = []
        for _, coco in t:
            if coco.value not in human.body_parts.keys():
                pose_2d_mpii.append((0, 0))
                visibilty.append(False)
                continue
            pose_2d_mpii.append((human.body_parts[coco.value].x, human.body_parts[coco.value].y))
            visibilty.append(True)
        return pose_2d_mpii, visibilty

MpiiColor=[[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

#convert kpts from opps to mpii
from_opps_converter={0:9, 1:8, 2:12, 3:11, 4:10, 5:13, 6:14, 7:15, 8:2, 9:1, 10:0, 11:3, 12:4, 13:5}
#convert kpts from mpii to opps
to_opps_converter={0:10, 1:9, 2:8, 3:11, 4:12, 5:13, 8:1, 9:0, 10:4, 11:3, 12:2, 13:5, 14:6, 15:7}

def opps_input_converter(mpii_kpts):
    cvt_kpts=np.zeros(shape=[16,2])
    transform = np.array([9,8,12,11,10,13,14,15,2,1,0,3,4,5])
    xs = mpii_kpts[0::3]
    ys = mpii_kpts[1::3]
    vs = mpii_kpts[2::3]
    lost_idx=np.where(vs<=0)[0]
    xs[lost_idx]=-1000
    ys[lost_idx]=-1000
    cvt_xs=xs[transform]
    cvt_ys=ys[transform]
    cvt_kpts[:-2,:]=np.array([cvt_xs,cvt_ys]).transpose()
    if(xs[2]<=0 or xs[3]<=0 or xs[12]<=0 or xs[13]<=0 or ys[2]<=0 or ys[3]<=0 or ys[12]<=0 or ys[13]<=0):
        center_x=-1000
        center_y=-1000
    else:
        center_x=(xs[2]+xs[3]+xs[12]+xs[13])/4
        center_y=(ys[2]+ys[3]+ys[12]+ys[13])/4
    cvt_kpts[14,:]=np.array([center_x,center_y])
    #adding background point
    cvt_kpts[-1:,:]=-1000
    return cvt_kpts

def opps_output_converter(kpt_list):
    kpts=[]
    mpii_keys=to_opps_converter.keys()
    for mpii_idx in range(0,16):
        if(mpii_idx in mpii_keys):
            model_idx=to_opps_converter[mpii_idx]
            x,y=kpt_list[model_idx]
            if(x<0 or y<0):
                kpts+=[0.0,0.0,-1.0]
            else:
                kpts+=[x,y,1.0]
        else:
            kpts+=[0.0,0.0,-1.0]
    return kpts

#convert kpts from ppn to mpii
from_ppn_converter={0:9, 1:8, 2:12, 3:11, 4:10, 5:13, 6:14, 7:15, 8:2, 9:1, 10:0, 11:3, 12:4, 13:5}
#convert kpts from mpii to ppn
to_ppn_converter={0:10, 1:9, 2:8, 3:11, 4:12, 5:13, 8:1, 9:0, 10:4, 11:3, 12:2, 13:5, 14:6, 15:7}

def ppn_input_converter(coco_kpts):
    cvt_kpts=np.zeros(shape=[16,2])
    transform = np.array([9,8,12,11,10,13,14,15,2,1,0,3,4,5])
    xs = coco_kpts[0::3]
    ys = coco_kpts[1::3]
    vs = coco_kpts[2::3]
    lost_idx=np.where(vs<=0)[0]
    xs[lost_idx]=-1000
    ys[lost_idx]=-1000
    cvt_xs=xs[transform]
    cvt_ys=ys[transform]
    cvt_kpts[:-2,:]=np.array([cvt_xs,cvt_ys]).transpose()
    center_x=(xs[2]+xs[3]+xs[12]+xs[13])/4
    center_y=(ys[2]+ys[3]+ys[12]+ys[13])/4
    cvt_kpts[14,:]=np.array([center_x,center_y])
    #adding person instance
    cvt_kpts[15,:]=(cvt_kpts[0,:]+cvt_kpts[1,:])/2
    return cvt_kpts

def ppn_output_converter(kpt_list):
    kpts=[]
    mpii_keys=to_ppn_converter.keys()
    for mpii_idx in range(0,16):
        if(mpii_idx in mpii_keys):
            model_idx=to_ppn_converter[mpii_idx]
            x,y=kpt_list[model_idx]
            if(x<0 or y<0):
                kpts+=[0.0,0.0,-1.0]
            else:
                kpts+=[x,y,1.0]
        else:
            kpts+=[0.0,0.0,-1.0]
    return kpts