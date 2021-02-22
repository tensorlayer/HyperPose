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
    
    def process(self,image,annos,mask_valid,bbxs=None):
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
            image, annos, mask_valid = tl.prepro.keypoint_random_flip(image, annos, mask_valid, prob=0.5, flip_list=self.flip_list)
        image, annos, mask_valid = tl.prepro.keypoint_resize_random_crop(image, annos, mask_valid, size=(self.hin, self.win))
        if(type(bbxs)==np.ndarray):
            #prepare transform bbx    
            transform_bbx=np.zeros(shape=(bbxs.shape[0],4,2))
            bbxs_x,bbxs_y,bbxs_w,bbxs_h=bbxs[:,0],bbxs[:,1],bbxs[:,2],bbxs[:,3]
            transform_bbx[:,0,0],transform_bbx[:,0,1]=bbxs_x,bbxs_y #left_top
            transform_bbx[:,1,0],transform_bbx[:,1,1]=bbxs_x+bbxs_w,bbxs_y #right_top
            transform_bbx[:,2,0],transform_bbx[:,2,1]=bbxs_x,bbxs_y+bbxs_h #left_buttom
            transform_bbx[:,3,0],transform_bbx[:,3,1]=bbxs_x+bbxs_w,bbxs_y+bbxs_h #right top
            transform_bbx=tl.prepro.affine_transform_keypoints(transform_bbx,transform_matrix)
            transform_bbx=np.array(transform_bbx)
            final_bbxs=np.zeros(shape=bbxs.shape)
            for bbx_id in range(0,transform_bbx.shape[0]):
                bbx=transform_bbx[bbx_id,:,:]
                bbx_min_x=np.amin(bbx[:,0])
                bbx_max_x=np.amax(bbx[:,0])
                bbx_min_y=np.amin(bbx[:,1])
                bbx_max_y=np.amax(bbx[:,1])
                final_bbxs[bbx_id,0]=bbx_min_x
                final_bbxs[bbx_id,1]=bbx_min_y
                final_bbxs[bbx_id,2]=bbx_max_x-bbx_min_x
                final_bbxs[bbx_id,3]=bbx_max_y-bbx_min_y
            resize_ratio=max(self.hin/image_h,self.win/image_w)
            final_bbxs[:,2]=final_bbxs[:,2]*resize_ratio
            final_bbxs[:,2]=final_bbxs[:,3]*resize_ratio
            bbxs=final_bbxs
            return image,annos,mask_valid,bbxs
        return image,annos,mask_valid
    