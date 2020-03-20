import os
import cv2
import math
import _pickle as cPickle
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer import logging
from tensorlayer.files.utils import (del_file, folder_exists, maybe_download_and_extract)

import matplotlib.pyplot as plt
from distutils.dir_util import mkpath
from scipy.spatial.distance import cdist
from pycocotools.coco import COCO, maskUtils

def get_dataset(config):
    if 'coco' in config.DATA.train_data:
        # automatically download MSCOCO data to "data/mscoco..."" folder
        train_im_path, train_ann_path, val_im_path, val_ann_path, _, _ = \
            load_mscoco_dataset(config.DATA.data_path, config.DATA.coco_version, task='person')

        # read coco training images contains valid people
        train_imgs_file_list, train_objs_info_list, train_mask_list, train_target_list = \
            get_pose_data_list(train_im_path, train_ann_path)

        # read coco validating images contains valid people (you can use it for training as well)
        val_imgs_file_list, val_objs_info_list, val_mask_list, val_target_list = \
            get_pose_data_list(val_im_path, val_ann_path)

    if 'custom' in config.DATA.train_data:
        ## read your own images contains valid people
        ## 1. if you only have one folder as follow:
        ##   data/your_data
        ##           /images
        ##               0001.jpeg
        ##               0002.jpeg
        ##           /coco.json
        # your_imgs_file_list, your_objs_info_list, your_mask_list, your_targets = \
        #     get_pose_data_list(config.DATA.your_images_path, config.DATA.your_annos_path)
        ## 2. if you have a folder with many folders: (which is common in industry)
        folder_list = tl.files.load_folder_list(path='data/your_data')
        your_imgs_file_list, your_objs_info_list, your_mask_list,your_target_list= [], [], [],[]
        for folder in folder_list:
            _imgs_file_list, _objs_info_list, _mask_list, _targets = \
                get_pose_data_list(os.path.join(folder, 'images'), os.path.join(folder, 'coco.json'))
            print(len(_imgs_file_list))
            your_imgs_file_list.extend(_imgs_file_list)
            your_objs_info_list.extend(_objs_info_list)
            your_mask_list.extend(_mask_list)
            your_target_list.extend(_targets)
        print("number of own images found:", len(your_imgs_file_list))

    # choose dataset for training
    if config.DATA.train_data == 'coco':
        # 1. only coco training set
        imgs_file_list = train_imgs_file_list
        train_target_list = train_target_list
    elif config.DATA.train_data == 'custom':
        # 2. only your own data
        imgs_file_list = your_imgs_file_list
        train_target_list = your_target_list
    elif config.DATA.train_data == 'coco_and_custom':
        # 3. your own data and coco training set
        imgs_file_list = train_imgs_file_list + your_imgs_file_list
        train_target_list = train_target_list+your_target_list
    else:
        raise Exception('please choose a valid config.DATA.train_data setting.')
    
    #tensorflow data pipeline
    def generator():
        """TF Dataset generator."""
        assert len(imgs_file_list) == len(train_target_list)
        for _input, _target in zip(imgs_file_list, train_target_list):
            yield _input.encode('utf-8'), cPickle.dumps(_target)

    train_dataset = tf.data.Dataset.from_generator(generator, output_types=(tf.string, tf.string))
    return train_dataset

## download dataset
def load_mscoco_dataset(path='data', dataset='2017', task='person'):  # TODO move to tl.files later
    """Download MSCOCO Dataset.
    Both 2014 and 2017 dataset have train, validate and test sets, but 2017 version put less data into the validation set (115k train, 5k validate) i.e. has more training data.

    Parameters
    -----------
    path : str
        The path that the data is downloaded to, defaults is ``data/mscoco...``.
    dataset : str
        The MSCOCO dataset version, `2014` or `2017`.
    task : str
        person for pose estimation, caption for image captioning, instance for segmentation.

    Returns
    ---------
    train_im_path : str
        Folder path of all training images.
    train_ann_path : str
        File path of training annotations.
    val_im_path : str
        Folder path of all validating images.
    val_ann_path : str
        File path of validating annotations.
    test_im_path : str
        Folder path of all testing images.
    test_ann_path : None
        File path of testing annotations, but as the test sets of MSCOCO 2014 and 2017 do not have annotation, returns None.

    Examples
    ----------
    >>> train_im_path, train_ann_path, val_im_path, val_ann_path, _, _ = \
    ...    tl.files.load_mscoco_dataset('data', '2017')

    References
    -------------
    - `MSCOCO <http://mscoco.org>`__.

    """
    import zipfile

    def unzip(path_to_zip_file, directory_to_extract_to):
        zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
        zip_ref.extractall(directory_to_extract_to)
        zip_ref.close()

    if dataset == "2014":
        logging.info("    [============= MSCOCO 2014 =============]")
        path = os.path.join(path, 'mscoco2014')

        if folder_exists(os.path.join(path, "annotations")) is False:
            logging.info("    downloading annotations")
            os.system("wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P {}".format(path))
            unzip(os.path.join(path, "annotations_trainval2014.zip"), path)
            del_file(os.path.join(path, "annotations_trainval2014.zip"))
        else:
            logging.info("    annotations exists")

        if folder_exists(os.path.join(path, "val2014")) is False:
            logging.info("    downloading validating images")
            os.system("wget http://images.cocodataset.org/zips/val2014.zip -P {}".format(path))
            unzip(os.path.join(path, "val2014.zip"), path)
            del_file(os.path.join(path, "val2014.zip"))
        else:
            logging.info("    validating images exists")

        if folder_exists(os.path.join(path, "train2014")) is False:
            logging.info("    downloading training images")
            os.system("wget http://images.cocodataset.org/zips/train2014.zip -P {}".format(path))
            unzip(os.path.join(path, "train2014.zip"), path)
            del_file(os.path.join(path, "train2014.zip"))
        else:
            logging.info("    training images exists")

        if folder_exists(os.path.join(path, "test2014")) is False:
            logging.info("    downloading testing images")
            os.system("wget http://images.cocodataset.org/zips/test2014.zip -P {}".format(path))
            unzip(os.path.join(path, "test2014.zip"), path)
            del_file(os.path.join(path, "test2014.zip"))
        else:
            logging.info("    testing images exists")
    elif dataset == "2017":
        # 11.5w train, 0.5w valid, test (no annotation)
        path = os.path.join(path, 'mscoco2017')

        if folder_exists(os.path.join(path, "annotations")) is False:
            logging.info("    downloading annotations")
            os.system("wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P {}".format(path))
            unzip(os.path.join(path, "annotations_trainval2017.zip"), path)
            del_file(os.path.join(path, "annotations_trainval2017.zip"))
        else:
            logging.info("    annotations exists")

        if folder_exists(os.path.join(path, "val2017")) is False:
            logging.info("    downloading validating images")
            os.system("wget http://images.cocodataset.org/zips/val2017.zip -P {}".format(path))
            unzip(os.path.join(path, "val2017.zip"), path)
            del_file(os.path.join(path, "val2017.zip"))
        else:
            logging.info("    validating images exists")

        if folder_exists(os.path.join(path, "train2017")) is False:
            logging.info("    downloading training images")
            os.system("wget http://images.cocodataset.org/zips/train2017.zip -P {}".format(path))
            unzip(os.path.join(path, "train2017.zip"), path)
            del_file(os.path.join(path, "train2017.zip"))
        else:
            logging.info("    training images exists")
        
        if folder_exists(os.path.join(path, "test2017")) is False:
            logging.info("    downloading testing images")
            os.system("wget http://images.cocodataset.org/zips/test2017.zip -P {}".format(path))
            unzip(os.path.join(path, "test2017.zip"), path)
            del_file(os.path.join(path, "test2017.zip"))
        else:
            logging.info("    testing images exists")
    else:
        raise Exception("dataset can only be 2014 and 2017, see MSCOCO website for more details.")

    # logging.info("    downloading annotations")
    # print(url, tar_filename)
    # maybe_download_and_extract(tar_filename, path, url, extract=True)
    # del_file(os.path.join(path, tar_filename))
    #
    # logging.info("    downloading images")
    # maybe_download_and_extract(tar_filename2, path, url2, extract=True)
    # del_file(os.path.join(path, tar_filename2))

    if dataset == "2014":
        train_images_path = os.path.join(path, "train2014")
        if task == "person":
            train_annotations_file_path = os.path.join(path, "annotations", "person_keypoints_train2014.json")
        elif task == "caption":
            train_annotations_file_path = os.path.join(path, "annotations", "captions_train2014.json")
        elif task == "instance":
            train_annotations_file_path = os.path.join(path, "annotations", "instances_train2014.json")
        else:
            raise Exception("unknown task")
        val_images_path = os.path.join(path, "val2014")
        if task == "person":
            val_annotations_file_path = os.path.join(path, "annotations", "person_keypoints_val2014.json")
        elif task == "caption":
            val_annotations_file_path = os.path.join(path, "annotations", "captions_val2014.json")
        elif task == "instance":
            val_annotations_file_path = os.path.join(path, "annotations", "instances_val2014.json")
        test_images_path = os.path.join(path, "test2014")
        test_annotations_file_path = None  #os.path.join(path, "annotations", "person_keypoints_test2014.json")
    else:
        train_images_path = os.path.join(path, "train2017")
        if task == "person":
            train_annotations_file_path = os.path.join(path, "annotations", "person_keypoints_train2017.json")
        elif task == "caption":
            train_annotations_file_path = os.path.join(path, "annotations", "captions_train2017.json")
        elif task == "instance":
            train_annotations_file_path = os.path.join(path, "annotations", "instances_train2017.json")
        else:
            raise Exception("unknown task")
        val_images_path = os.path.join(path, "val2017")
        if task == "person":
            val_annotations_file_path = os.path.join(path, "annotations", "person_keypoints_val2017.json")
        elif task == "caption":
            val_annotations_file_path = os.path.join(path, "annotations", "captions_val2017.json")
        elif task == "instance":
            val_annotations_file_path = os.path.join(path, "annotations", "instances_val2017.json")
        test_images_path = os.path.join(path, "test2017")
        test_annotations_file_path = None  #os.path.join(path, "annotations", "person_keypoints_test2017.json")
    return train_images_path, train_annotations_file_path, \
            val_images_path, val_annotations_file_path, \
                test_images_path, test_annotations_file_path


def get_pose_data_list(im_path, ann_path):
    """
    train_im_path : image folder name
    train_ann_path : coco json file name
    """
    print("[x] Get pose data from {}".format(im_path))
    data = PoseInfo(im_path, ann_path, False)
    imgs_file_list = data.get_image_list()
    objs_info_list = data.get_joint_list()
    mask_list = data.get_mask()
    bbx_list=data.get_bbx_list()
    targets = list(zip(objs_info_list, mask_list,bbx_list))
    if len(imgs_file_list) != len(objs_info_list):
        raise Exception("number of images and annotations do not match")
    else:
        print("{} has {} images".format(im_path, len(imgs_file_list)))
    return imgs_file_list, objs_info_list, mask_list, targets

## read coco data
class CocoMeta:
    """ Be used in PoseInfo. """
    limb = list(
        zip([2, 9, 10, 2, 12, 13, 2, 3, 4, 3, 2, 6, 7, 6, 2, 1, 1, 15, 16],
            [9, 10, 11, 12, 13, 14, 3, 4, 5, 17, 6, 7, 8, 18, 1, 15, 16, 17, 18]))

    def __init__(self, idx, img_url, img_meta, annotations, masks, bbxs):
        self.idx = idx
        self.img_url = img_url
        self.img = None
        self.height = int(img_meta['height'])
        self.width = int(img_meta['width'])
        self.masks = masks
        self.bbx_list=bbxs
        joint_list = []

        for anno in annotations:
            if anno.get('num_keypoints', 0) == 0:
                continue

            kp = np.array(anno['keypoints'])
            xs = kp[0::3]
            ys = kp[1::3]
            vs = kp[2::3]
            # if joint is marked
            joint_list.append([[x, y] if v >= 1 else (-1000, -1000) for x, y, v in zip(xs, ys, vs)])

        self.joint_list = []
        # 对原 COCO 数据集的转换 其中第二位之所以不一样是为了计算 Neck 等于左右 shoulder 的中点
        transform = list(
            zip([1, 6, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4],
                [1, 7, 7, 9, 11, 6, 8, 10, 13, 15, 17, 12, 14, 16, 3, 2, 5, 4]))
        for prev_joint in joint_list:
            new_joint = []
            for idx1, idx2 in transform:
                j1 = prev_joint[idx1 - 1]
                j2 = prev_joint[idx2 - 1]

                if j1[0] <= 0 or j1[1] <= 0 or j2[0] <= 0 or j2[1] <= 0:
                    new_joint.append([-1000, -1000])
                else:
                    new_joint.append([(j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2])

            # for background
            new_joint.append([-1000, -1000])
            if len(new_joint) != 19:
                print('The Length of joints list should be 0 or 19 but actually:', len(new_joint))
            self.joint_list.append(new_joint)


class PoseInfo:
    """ Use COCO for pose estimation, returns images with people only. """

    def __init__(self, image_base_dir, anno_path, with_mask):
        self.metas = []
        # self.data_dir = data_dir
        # self.data_type = data_type
        self.image_base_dir = image_base_dir
        self.anno_path = anno_path
        self.with_mask = with_mask
        self.coco = COCO(self.anno_path)
        self.get_image_annos()
        self.image_list = os.listdir(self.image_base_dir)

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

            images_info = self.coco.loadImgs(images_ids[idx])
            image_path = os.path.join(self.image_base_dir, images_info[0]['file_name'])
            # filter that some images might not in the list
            if not os.path.exists(image_path):
                print("[skip] json annotation found, but cannot found image: {}".format(image_path))
                continue

            annos_ids = self.coco.getAnnIds(imgIds=images_ids[idx])
            annos_info = self.coco.loadAnns(annos_ids)
            keypoints = self.get_keypoints(annos_info)
            bbxs=self.get_bbxs(annos_info)

            #############################################################################
            anns = annos_info
            prev_center = []
            masks = []

            # sort from the biggest person to the smallest one
            if self.with_mask:
                persons_ids = np.argsort([-a['area'] for a in anns], kind='mergesort')

                for p_id in list(persons_ids):
                    person_meta = anns[p_id]

                    if person_meta["iscrowd"]:
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
            total_keypoints = sum([ann.get('num_keypoints', 0) for ann in annos_info])
            if total_keypoints > 0:
                meta = CocoMeta(images_ids[idx], image_path, images_info[0], keypoints, masks, bbxs)
                self.metas.append(meta)

        print("Overall get {} valid pose images from {} and {}".format(
            len(self.metas), self.image_base_dir, self.anno_path))

    def load_images(self):
        pass

    def get_image_list(self):
        img_list = []
        for meta in self.metas:
            img_list.append(meta.img_url)
        return img_list

    def get_joint_list(self):
        joint_list = []
        for meta in self.metas:
            joint_list.append(meta.joint_list)
        return joint_list

    def get_mask(self):
        mask_list = []
        for meta in self.metas:
            mask_list.append(meta.masks)
        return mask_list
    
    def get_bbx_list(self):
        bbx_list=[]
        for meta in self.metas:
            bbx_list.append(meta.bbx_list)
        return bbx_list

if __name__ == '__main__':
    data_dir = '/Users/Joel/Desktop/coco'
    data_type = 'val'
    anno_path = '{}/annotations/person_keypoints_{}2014.json'.format(data_dir, data_type)
    df_val = PoseInfo(data_dir, data_type, anno_path)

    for i in range(50):
        meta = df_val.metas[i]
        mask_sig = meta.masks
        print('shape of np mask is ', np.shape(mask_sig), type(mask_sig))
        if mask_sig is not []:
            mask_miss = np.ones((meta.height, meta.width), dtype=np.uint8)
            for seg in mask_sig:
                bin_mask = maskUtils.decode(seg)
                bin_mask = np.logical_not(bin_mask)
                mask_miss = np.bitwise_and(mask_miss, bin_mask)
