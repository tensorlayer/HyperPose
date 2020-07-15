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