import os
import numpy as np
from pycocotools.coco import COCO
from scipy.spatial.distance import cdist

## read coco data
class CocoMeta:
    """ Be used in PoseInfo. """
    def __init__(self, image_id, img_url, img_meta, kpts_infos, masks, bbxs, is_crowd):
        self.image_id = image_id
        self.img_url = img_url
        self.img = None
        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])
        self.masks = masks
        self.bbx_list=bbxs
        self.is_crowd=is_crowd

        self.joint_list=[]
        for kpts_info in kpts_infos:
            if kpts_info.get('num_keypoints', 0) == 0:
                continue
            kpts = np.array(kpts_info['keypoints'])
            self.joint_list.append(kpts)

class PoseInfo:
    """ Use COCO for pose estimation, returns images with people only. """

    def __init__(self, image_base_dir, anno_path, with_mask=True,dataset_filter=None, eval=False):
        self.metas = []
        # self.data_dir = data_dir
        # self.data_type = data_type
        self.eval=eval
        self.image_base_dir = image_base_dir
        self.anno_path = anno_path
        self.with_mask = with_mask
        self.coco = COCO(self.anno_path)
        self.get_image_annos()
        self.image_list = os.listdir(self.image_base_dir)
        if(dataset_filter!=None):
            filter_metas=[]
            for meta in self.metas:
                if(dataset_filter(meta)==True):
                    filter_metas.append(meta)
            self.metas=filter_metas

    @staticmethod
    def get_keypoints(annos_info):
        annolist = []
        for anno in annos_info:
            adjust_anno = {'keypoints': anno['keypoints'], 'num_keypoints': anno['num_keypoints']}
            annolist.append(adjust_anno)
        return annolist

    @staticmethod    
    def get_bbxs(annos_info):
        bbxlist=[]
        for anno in annos_info:
            bbxlist.append(anno["bbox"])
        return bbxlist

    def get_image_annos(self):
        """Read JSON file, and get and check the image list.
        Skip missing images.
        """
        images_ids = self.coco.getImgIds()
        len_imgs = len(images_ids)
        for idx in range(len_imgs):

            image_info = self.coco.loadImgs(images_ids[idx])[0]
            image_path = os.path.join(self.image_base_dir, image_info['file_name'])
            # filter that some images might not in the list
            if not os.path.exists(image_path):
                print("[skip] json annotation found, but cannot found image: {}".format(image_path))
                continue

            annos_ids = self.coco.getAnnIds(imgIds=images_ids[idx])
            annos_info = self.coco.loadAnns(annos_ids)
            kpts_info = self.get_keypoints(annos_info)
            bbxs=self.get_bbxs(annos_info)

            #############################################################################
            anns = annos_info
            prev_center = []
            masks = []
            #check for crowd
            is_crowd=False
            for ann in anns:
                if("iscrowd" in ann and ann["iscrowd"]):
                    is_crowd=True

            # sort from the biggest person to the smallest one
            if self.with_mask:
                persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

                for p_id in list(persons_ids):
                    person_meta = anns[p_id]

                    if person_meta["iscrowd"]:
                        is_crowd=True
                        masks.append(self.coco.annToRLE(person_meta))
                        continue

                    # skip this person if parts number is too low or if
                    # segmentation area is too small
                    if person_meta["num_keypoints"] < 5 or person_meta["area"] < 32 * 32:
                        masks.append(self.coco.annToRLE(person_meta))
                        continue

                    person_center = [
                        person_meta["bbox"][0] + person_meta["bbox"][2] / 2,
                        person_meta["bbox"][1] + person_meta["bbox"][3] / 2
                    ]

                    # skip this person if the distance to existing person is too small
                    too_close = False
                    for pc in prev_center:
                        a = np.expand_dims(pc[:2], axis=0)
                        b = np.expand_dims(person_center, axis=0)
                        dist = cdist(a, b)[0]
                        if dist < pc[2] * 0.3:
                            too_close = True
                            break

                    if too_close:
                        # add mask of this person. we don't want to show the network
                        # unlabeled people
                        masks.append(self.coco.annToRLE(person_meta))
                        continue

            ############################################################################
            #eval accept all images
            if(self.eval):
                meta = CocoMeta(images_ids[idx], image_path, image_info, kpts_info, masks, bbxs, is_crowd)
                self.metas.append(meta)
            #train filter images
            else:
                total_keypoints = sum([ann.get('num_keypoints', 0) for ann in annos_info])
                if total_keypoints > 0:
                    meta = CocoMeta(images_ids[idx], image_path, image_info, kpts_info, masks, bbxs, is_crowd)
                    self.metas.append(meta)

        print("Overall get {} valid pose images from {} and {}".format(
            len(self.metas), self.image_base_dir, self.anno_path))

    def load_images(self):
        pass

    def get_image_id_list(self):
        img_id_list=[]
        for meta in self.metas:
            img_id_list.append(meta.image_id)
        return img_id_list

    def get_image_list(self):
        img_list = []
        for meta in self.metas:
            img_list.append(meta.img_url)
        return img_list

    def get_kpt_list(self):
        joint_list = []
        for meta in self.metas:
            joint_list.append(meta.joint_list)
        return joint_list

    def get_mask_list(self):
        mask_list = []
        for meta in self.metas:
            mask_list.append(meta.masks)
        return mask_list
    
    def get_bbx_list(self):
        bbx_list=[]
        for meta in self.metas:
            bbx_list.append(meta.bbx_list)
        return bbx_list
