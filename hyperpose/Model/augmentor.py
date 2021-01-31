import cv2
import numpy as np
import tensorflow as tf
import tensorlayer as tl

class Augmentor:
    def __init__(self,hin,win,angle_min=-30,angle_max=30,zoom_min=0.5,zoom_max=0.8,flip_list=None):
        self.hin=hin
        self.win=win
        self.angle_min=angle_min
        self.angle_max=angle_max
        self.zoom_min=zoom_min
        self.zoom_max=zoom_max
        self.flip_list=flip_list
    
    def process(image,annos,mask_valid):
        #get transform matrix
        image_h,image_w,_=image.shape
        M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
        M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
        M_combined = M_rotate.dot(M_zoom)
        transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=image_w, y=image_h)
        #apply data augmentation
        image = tl.prepro.affine_transform_cv2(image, transform_matrix)
        annos = tl.prepro.affine_transform_keypoints(annos, transform_matrix)
        mask_valid = tl.prepro.affine_transform_cv2(mask_valid, transform_matrix, border_mode='replicate')
        if(self.flip_list!=None):
            image, annos, mask_valid = tl.prepro.keypoint_random_flip(image, annos, mask_valid, prob=0.5, flip_list=flip_list)
        image, annos, mask_valid = tl.prepro.keypoint_resize_random_crop(image, annos, mask_valid, size=(self.hin, self.win))
        return image,annos,mask_valid
    