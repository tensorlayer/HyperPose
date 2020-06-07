from enum import Enum

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