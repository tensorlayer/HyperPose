import os
import json
import numpy as np
import scipy
from scipy import io

class MPIIMeta:
    def __init__(self,image_path,annos_list):
        #print(f"test get meta with img_path:{image_path} annos_list:{annos_list}\n\n")
        image_name=os.path.basename(image_path)
        self.image_id=int(image_name[:image_name.index(".")])
        self.image_path=image_path
        self.n_pos=16
        self.annos_list=annos_list
        self.headbbx_list=[]
        self.scale_list=[]
        self.center_list=[]
        self.joint_list=[]
        for anno in self.annos_list:
            #head bbx
            x1,y1,x2,y2=anno["x1"],anno["y1"],anno["x2"],anno["y2"]
            center_x=(x1+x2)/2
            center_y=(y1+y2)/2
            w,h=x2-x1,y2-y1
            headbbx=np.array([center_x,center_y,w,h]).astype(np.float32)
            self.headbbx_list.append(headbbx)
            #scale
            scale=anno["scale"]
            self.scale_list.append(np.float32(scale*200.0))
            #pos
            pos_x,pos_y=anno["pos_x"],anno["pos_y"]
            pos=np.array([pos_x,pos_y]).astype(np.float32)
            self.center_list.append(pos)
            #kpts
            kpts=[]
            for kpt_id in range(0,self.n_pos):
                x,y,v=anno["kpts"][str(kpt_id)]
                kpts+=[x,y,v]
            self.joint_list.append(np.array(kpts))
    
    def to_anns_list(self):
        anns_list=[]
        for headbbx,scale,center,joints in zip(self.headbbx_list,self.scale_list,self.center_list,self.joint_list):
            ann_dict={}
            ann_dict["headbbx"]=headbbx
            ann_dict["scale"]=scale
            ann_dict["center"]=center
            vis=joints[2::3]
            ann_dict["keypoints"]=joints
            ann_dict["vis"]=vis
            anns_list.append(ann_dict)
        return anns_list

class PoseInfo:
    def __init__(self,image_dir,annos_path,dataset_filter=None):
        self.metas=[]
        self.n_pos=16
        self.image_dir=image_dir
        self.annos_path=annos_path
        self.get_image_annos()
        if(dataset_filter!=None):
            filter_metas=[]
            for meta in self.metas:
                if(dataset_filter(meta)==True):
                    filter_metas.append(meta)
            self.metas=filter_metas
    
    def get_image_annos(self):
        json_dict=json.load(open(self.annos_path,"r"))
        for image_path in json_dict.keys():
            annos_list=json_dict[image_path]
            self.metas.append(MPIIMeta(os.path.join(self.image_dir,image_path),annos_list))

    def get_image_id_list(self):
        image_id_list=[]
        for meta in self.metas:
            image_id_list.append(meta.image_id)
        return image_id_list

    def get_image_list(self):
        image_list=[]
        for meta in self.metas:
            image_list.append(meta.image_path)
        return image_list
    
    def get_headbbx_list(self):
        headbbx_list=[]
        for meta in self.metas:
            headbbx_list.append(meta.headbbx_list)
        return headbbx_list
    
    def get_scale_list(self):
        scale_list=[]
        for meta in self.metas:
            scale_list.append(meta.scale_list)
        return scale_list
    
    def get_center_list(self):
        pos_list=[]
        for meta in self.metas:
            pos_list.append(meta.center_list)
        return pos_list 

    def get_kpt_list(self):
        joint_list=[]
        for meta in self.metas:
            joint_list.append(meta.joint_list)
        return joint_list



def generate_json(mat_path,is_test=False):
    #utils
    def check_exist(mat_obj,field_name):
        if(field_name in mat_obj._fieldnames):
            if(0 not in mat_obj.__dict__[field_name].shape):
                return True
        return False

    #init anno_dict in case for lost annotation
    def get_init_dict():
        anno_dict={}
        #init person info
        anno_names=["x1","y1","x2","y2","scale","pos_x","pos_y"]
        for anno_name in anno_names:
            anno_dict[anno_name]=-1.0
        #init kpts
        anno_dict["kpts"]={}
        for kpt_id in range(0,16):
            anno_dict["kpts"][kpt_id]=[-1000.0,-1000.0,-1]
        return anno_dict

    json_dict={}
    mat=io.loadmat(mat_path,struct_as_record=False)
    data_obj=mat["RELEASE"][0][0]
    anno_list=data_obj.__dict__["annolist"][0]
    train_marks=data_obj.__dict__["img_train"][0]
    if(is_test):
        target_idx=np.where(train_marks==0)
    else:
        target_idx=np.where(train_marks==1)
    target_anno_list=anno_list[target_idx]
    for anno in target_anno_list:
        #get image path
        image_obj=anno.__dict__["image"][0][0]
        image_path=image_obj.__dict__["name"][0]
        #get annotations
        annos_list=[]
        if(len(anno.__dict__["annorect"])==0):
            continue
        anno_objs=anno.__dict__["annorect"][0]
        #handle each person annotation
        for anno_obj in anno_objs:
            anno_dict=get_init_dict()
            #get bbx and scale
            for exp_name in ["x1","y1","x2","y2","scale"]:
                if(check_exist(anno_obj,exp_name)):
                    anno_dict[exp_name]=float(anno_obj.__dict__[exp_name][0][0])
            #get pose
            if(check_exist(anno_obj,"objpos")):
                pos_obj=anno_obj.__dict__["objpos"][0][0]
                if(check_exist(pos_obj,"x")):
                    anno_dict["pos_x"]=float(pos_obj.__dict__["x"][0][0])
                if(check_exist(pos_obj,"y")):
                    anno_dict["pos_y"]=float(pos_obj.__dict__["y"][0][0])   
            #get kpts
            if(check_exist(anno_obj,"annopoints")):
                anno_points_obj=anno_obj.__dict__["annopoints"][0][0]
                if(check_exist(anno_points_obj,"point")):
                    kpt_objs=anno_points_obj.__dict__["point"][0]
                    for kpt_obj in kpt_objs:
                        if(check_exist(kpt_obj,"id")):
                            kpt_id=int(kpt_obj.__dict__["id"][0][0])
                        if(check_exist(kpt_obj,"x")):
                            kpt_x=float(kpt_obj.__dict__["x"][0][0])
                        if(check_exist(kpt_obj,"y")):
                            kpt_y=float(kpt_obj.__dict__["y"][0][0])
                        if(check_exist(kpt_obj,"is_visible")):
                            kpt_v=kpt_obj.__dict__["is_visible"]
                        if(type(kpt_v)==float):
                            kpt_v=float(kpt_v)
                        elif(type(kpt_v)==np.ndarray):
                            if(len(kpt_v.shape)==1):
                                kpt_v=float(kpt_v[0])
                            elif(len(kpt_v.shape)==2):
                                kpt_v=float(kpt_v[0][0])
                        #plus 1 for the difference between visible definition of coco and mpii
                        anno_dict["kpts"][int(kpt_id)]=[kpt_x,kpt_y,kpt_v+1]
            annos_list.append(anno_dict)
        if(image_path not in json_dict):
            json_dict[image_path]=[]
        json_dict[image_path]+=annos_list
    return json_dict