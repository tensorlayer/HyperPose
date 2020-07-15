import cv2
import numpy as np
import zipfile
from enum import Enum
from ..Config.define import TRAIN,MODEL,DATA,KUNGFU

def unzip(path_to_zip_file, directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

def imread_rgb_float(image_path,data_format="channels_first"):
    image=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    if(data_format=="channels_first"):
        image=np.transpose(image,[2,0,1])
    return image.copy()

def imwrite_rgb_float(image,image_path,data_format="channels_first"):
    if(data_format=="channels_first"):
        image=np.transpose(image,[1,2,0])
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image=np.clip(image*255.0,0,255).astype(np.uint8)
    return cv2.imwrite(image_path,image)
