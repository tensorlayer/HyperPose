import numpy as np
from enum import Enum

#specialize for coco
class CocoPart(Enum):
    Nose=0
    LEye=1
    REye=2
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

CocoLimb=[[15, 13],[13, 11],[16, 14],[14, 12],[11, 12],[ 5, 11],[ 6, 12],[ 5,  6],[ 5,  7],\
    [ 6,  8],[ 7,  9],[ 8, 10],[ 1,  2],[ 0,  1],[ 0,  2],[ 1,  3],[ 2,  4],[ 3,  5],[ 4,  6]]

CocoColor = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
COCO_SIGMA=[
    0.026,  # nose
    0.025,  # eyes
    0.025,  # eyes
    0.035,  # ears
    0.035,  # ears
    0.079,  # shoulders
    0.079,  # shoulders
    0.072,  # elbows
    0.072,  # elbows
    0.062,  # wrists
    0.062,  # wrists
    0.107,  # hips
    0.107,  # hips
    0.087,  # knees
    0.087,  # knees
    0.089,  # ankles
    0.089,  # ankles
]

COCO_UPRIGHT_POSE = np.array([
    [0.0, 9.3, 2.0],  # 'nose',            # 1
    [-0.35, 9.7, 2.0],  # 'left_eye',        # 2
    [0.35, 9.7, 2.0],  # 'right_eye',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
    [1.4, 2.1, 2.0],  # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
])
area_ref=((np.max(COCO_UPRIGHT_POSE[:, 0]) - np.min(COCO_UPRIGHT_POSE[:, 0])) *
            (np.max(COCO_UPRIGHT_POSE[:, 1]) - np.min(COCO_UPRIGHT_POSE[:, 1])))

c, s = np.cos(np.deg2rad(45)), np.sin(np.deg2rad(45))
rotate = np.array(((c, -s), (s, c)))
COCO_UPRIGHT_POSE_45=np.einsum('ij,kj->ki', rotate, np.copy(COCO_UPRIGHT_POSE[:,:2]))
area_ref_45=((np.max(COCO_UPRIGHT_POSE_45[:, 0]) - np.min(COCO_UPRIGHT_POSE_45[:, 0])) *
            (np.max(COCO_UPRIGHT_POSE_45[:, 1]) - np.min(COCO_UPRIGHT_POSE_45[:, 1])))

#specialize for MPII
#TODO: modified to be specialized for MPII
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

