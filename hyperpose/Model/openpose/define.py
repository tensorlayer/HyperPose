import numpy as np
from enum import Enum
#specialized for coco
class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

CocoLimb=list(zip([1, 8, 9,  1,  11, 12, 1, 2, 3,  2, 1, 5, 6, 5,  1,  0,  0,  14, 15],
                  [8, 9, 10, 11, 12, 13, 2, 3, 4, 16, 5, 6, 7, 17, 0, 14, 15,  16, 17]))

CocoColor = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

to_coco_converter={0:0, 2:6, 3:8, 4:10, 5:5, 6:7, 7:9, 8:12, 9:14, 10:16, 11:11, 12:13, 13:15, 14:2, 15:1, 16:4, 17:3}

from_coco_converter={0:0, 1:15, 2:14, 3:17, 4:16, 5:5, 6:2, 7:6, 8:3, 9:7, 10:4, 11:11, 12:8, 13:12, 14:9, 15:13, 16:10}

def coco_input_converter(coco_kpts):
    cvt_kpts=np.zeros(shape=[len(CocoPart),2])
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

def coco_output_converter(kpt_list):
    kpts=[]
    for coco_idx in list(from_coco_converter.keys()):
        model_idx=from_coco_converter[coco_idx]
        x,y=kpt_list[model_idx]
        if(x<0 or y<0):
            kpts+=[0.0,0.0,0.0]
        else:
            kpts+=[x,y,2.0]
    return kpts

def get_coco_flip_list():
    flip_list=[]
    for part_idx,part in enumerate(CocoPart):
        #eye flip
        if(part==CocoPart.REye):
            flip_list.append(CocoPart.LEye.value)
        elif(part==CocoPart.LEye):
            flip_list.append(CocoPart.REye.value)
        #ear flip
        elif(part==CocoPart.REar):
            flip_list.append(CocoPart.LEar.value)
        elif(part==CocoPart.LEar):
            flip_list.append(CocoPart.REar.value)
        #shoulder flip
        elif(part==CocoPart.RShoulder):
            flip_list.append(CocoPart.LShoulder.value)
        elif(part==CocoPart.LShoulder):
            flip_list.append(CocoPart.RShoulder.value)
        #elbow flip
        elif(part==CocoPart.RElbow):
            flip_list.append(CocoPart.LElbow.value)
        elif(part==CocoPart.LElbow):
            flip_list.append(CocoPart.RElbow.value)
        #wrist flip
        elif(part==CocoPart.RWrist):
            flip_list.append(CocoPart.LWrist.value)
        elif(part==CocoPart.LWrist):
            flip_list.append(CocoPart.RWrist.value)
        #hip flip
        elif(part==CocoPart.RHip):
            flip_list.append(CocoPart.LHip.value)
        elif(part==CocoPart.LHip):
            flip_list.append(CocoPart.RHip.value)
        #knee flip
        elif(part==CocoPart.RKnee):
            flip_list.append(CocoPart.LKnee.value)
        elif(part==CocoPart.LKnee):
            flip_list.append(CocoPart.RKnee.value)
        #ankle flip
        elif(part==CocoPart.RAnkle):
            flip_list.append(CocoPart.LAnkle.value)
        elif(part==CocoPart.LAnkle):
            flip_list.append(CocoPart.RAnkle.value)
        #others
        else:
            flip_list.append(part.value)
    return flip_list

Coco_flip_list=get_coco_flip_list()

#specialized for mpii
class MpiiPart(Enum):
    Headtop=0
    Neck=1
    RShoulder=2
    RElbow=3
    RWrist=4
    LShoulder=5
    LElbow=6
    LWrist=7
    RHip=8
    RKnee=9
    RAnkle=10
    LHip=11
    LKnee=12
    LAnkle=13
    Center=14
    Background=15

MpiiLimb=list(zip([0, 1, 2, 3, 1, 5, 6, 1,  14,  8, 9,  14, 11, 12],
                  [1, 2, 3, 4, 5, 6, 7, 14,  8,  9, 10, 11, 12, 13]))

MpiiColor = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

to_mpii_converter={0:9, 1:8, 2:12, 3:11, 4:10, 5:13, 6:14, 7:15, 8:2, 9:1, 10:0, 11:3, 12:4, 13:5}

from_mpii_converter={0:10, 1:9, 2:8, 3:11, 4:12, 5:13, 8:1, 9:0, 10:4, 11:3, 12:2, 13:5, 14:6, 15:7}

def mpii_input_converter(mpii_kpts):
    cvt_kpts=np.zeros(shape=[len(MpiiPart),2])
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

def mpii_output_converter(kpt_list):
    kpts=[]
    mpii_keys=from_mpii_converter.keys()
    for mpii_idx in range(0,16):
        if(mpii_idx in mpii_keys):
            model_idx=from_mpii_converter[mpii_idx]
            x,y=kpt_list[model_idx]
            if(x<0 or y<0):
                kpts+=[0.0,0.0,-1.0]
            else:
                kpts+=[x,y,1.0]
        else:
            kpts+=[0.0,0.0,-1.0]
    return kpts

def get_mpii_flip_list():
    flip_list=[]
    for part_idx,part in enumerate(MpiiPart):
        #shoulder flip
        if(part==MpiiPart.RShoulder):
            flip_list.append(MpiiPart.LShoulder.value)
        elif(part==MpiiPart.LShoulder):
            flip_list.append(MpiiPart.RShoulder.value)
        #elbow flip
        elif(part==MpiiPart.RElbow):
            flip_list.append(MpiiPart.LElbow.value)
        elif(part==MpiiPart.LElbow):
            flip_list.append(MpiiPart.RElbow.value)
        #wrist flip
        elif(part==MpiiPart.RWrist):
            flip_list.append(MpiiPart.LWrist.value)
        elif(part==MpiiPart.LWrist):
            flip_list.append(MpiiPart.RWrist.value)
        #hip flip
        elif(part==MpiiPart.RHip):
            flip_list.append(MpiiPart.LHip.value)
        elif(part==MpiiPart.LHip):
            flip_list.append(MpiiPart.RHip.value)
        #knee flip
        elif(part==MpiiPart.RKnee):
            flip_list.append(MpiiPart.LKnee.value)
        elif(part==MpiiPart.LKnee):
            flip_list.append(MpiiPart.RKnee.value)
        #ankle flip
        elif(part==MpiiPart.RAnkle):
            flip_list.append(MpiiPart.LAnkle.value)
        elif(part==MpiiPart.LAnkle):
            flip_list.append(MpiiPart.RAnkle.value)
        #others
        else:
            flip_list.append(part.value)
    return flip_list

Mpii_flip_list=get_mpii_flip_list()