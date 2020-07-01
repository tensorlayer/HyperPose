from enum import Enum
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