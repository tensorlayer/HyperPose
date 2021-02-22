import cv2

class Human:
    """
    body_parts: list of BodyPart
    """

    def __init__(self,parts,limbs,colors):
        self.local_id=-1
        self.global_id=-1
        self.parts=parts
        self.limbs=limbs
        self.colors=colors
        self.body_parts = {}
        self.score = 0.0
        self.bbx=None
        self.area=None
    
    def get_global_id(self):
        return int(self.global_id)
    
    def get_score(self):
        for part_idx in self.body_parts.keys():
            body_part=self.body_parts[part_idx]
            self.score+=body_part.score
        self.score=self.score/len(self.body_parts.keys())
        return float(self.score)
    
    def get_partnum(self):
        return len(self.body_parts.keys())
    
    def get_bbx(self):
        min_x,min_y=10000,10000
        max_x,max_y=-1,-1
        for body_part_idx in self.body_parts.keys():
            body_part=self.body_parts[body_part_idx]
            x=body_part.x
            y=body_part.y
            min_x=min(x,min_x)
            min_y=min(y,min_y)
            max_x=max(x,max_x)
            max_y=max(y,max_y)
        center_x=(min_x+max_x)/2
        center_y=(min_y+max_y)/2
        h=max_y-min_y
        w=max_x-min_x
        self.bbx=[center_x,center_y,w,h]
        return [center_x,center_y,w,h]
    
    def get_area(self):
        bbx=self.get_bbx()
        self.area=float(bbx[2]*bbx[3])
        return self.area
    
    def bias(self,bias_w,bias_h):
        for part_idx in self.body_parts.keys():
            body_part=self.body_parts[part_idx]
            body_part.x=body_part.x+bias_w
            body_part.y=body_part.y+bias_h
            if(body_part.x<0):
                body_part.x=-1000.0
            if(body_part.y<0):
                body_part.y=-1000.0
    
    def scale(self,scale_w,scale_h):
        for part_idx in self.body_parts.keys():
            body_part=self.body_parts[part_idx]
            body_part.x=body_part.x*scale_w
            body_part.y=body_part.y*scale_h
            body_part.w=body_part.w*scale_w
            body_part.h=body_part.h*scale_h
            if(body_part.x<0):
                body_part.x=-1000.0
            if(body_part.y<0):
                body_part.y=-1000.0
                
    
    def draw_human(self,img):
        img_h,img_w,img_c=img.shape
        radius=int(min(img_h,img_w)/80)
        thickness=int(min(img_h,img_w)/100)
        for part_idx in self.body_parts.keys():
            body_part=self.body_parts[part_idx]
            x=body_part.x
            y=body_part.y
            if(x<0 or x>=img_w or y<0 or y>=img_h):
                continue
            color=self.colors[part_idx]
            img=cv2.circle(img,(int(x),int(y)),radius=radius,color=color,thickness=-1)
        line_color=(255,0,0)
        for limb in self.limbs:
            src_part_idx,dst_part_idx=limb
            if((src_part_idx in self.body_parts) and (dst_part_idx in self.body_parts)):
                src_body_part=self.body_parts[src_part_idx]
                src_x,src_y=int(src_body_part.x),int(src_body_part.y)
                dst_body_part=self.body_parts[dst_part_idx]
                dst_x,dst_y=int(dst_body_part.x),int(dst_body_part.y)
                if(src_x<0 or src_x>=img_w or src_y<0 or src_y>=img_h):
                    continue
                if(dst_x<0 or dst_x>=img_w or dst_y<0 or dst_y>=img_h):
                    continue
                img=cv2.line(img,(src_x,src_y),(dst_x,dst_y),color=line_color,thickness=thickness)
        return img
    
    def print(self):
        for part_idx in self.body_parts.keys():
            body_part=self.body_parts[part_idx]
            print(f"body-part:{self.parts(part_idx)} x:{body_part.x} y:{body_part.y} score:{body_part.score}")
        print()

    def __str__(self):
        return ' '.join([str(x) for x in self.body_parts.values()])

    def __repr__(self):
        return self.__str__()

class BodyPart:
    """
    part_idx : part index(eg. 0 for nose)
    x, y: coordinate of body part
    score : confidence score
    """

    def __init__(self, parts, u_idx, part_idx, x, y, score, w=-1, h=-1 ):
        self.parts=parts
        self.u_idx=u_idx
        self.part_idx = part_idx
        self.x, self.y = x, y
        self.w, self.h = w, h
        self.score = score

    def get_part_name(self):
        return self.parts(self.part_idx)
    
    def get_x(self):
        return float(self.x)
    
    def get_y(self):
        return float(self.y)

    def __str__(self):
        return 'BodyPart:%d-(%.2f, %.2f) score=%.2f' % (self.part_idx, self.x, self.y, self.score)

    def __repr__(self):
        return self.__str__()