import cv2
import numpy as np
import matplotlib.pyplot as plt
from .common import pad_image_shape
from .common import image_float_to_uint8

# Basic class of processors to be inherit
class BasicPreProcessor:
    def __init__(self, parts, limbs, hin, win, hout, wout, colors=None, data_format="channels_first", *args, **kargs):
        self.parts = parts
        self.limbs = limbs
        self.hin, self.win = hin, win
        self.hout, self.wout = hout, wout
        self.colors = colors
        self.data_format = data_format
    
    def process(self, annos, mask, bbxs):
        raise NotImplementedError("abstract class BasicPreProcessor function: process not implemented!")

class BasicPostProcessor:
    def __init__(self, parts, limbs, colors=None, data_format="channels_first", *args, **kargs):
        self.parts = parts
        self.limbs = limbs
        self.colors = colors
        self.data_format = data_format
    
    def process(self, predict_x):
        raise NotImplementedError("abstract class BasicPostProcessor function: process not implemented!")

class BasicVisualizer:
    def __init__(self, save_dir="./save_dir", *args, **kargs):
        self.save_dir = save_dir
    
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
    
    def visualize_result(self, image, humans, name):
        pltdrawer = PltDrawer(draw_row=1, draw_col=2, figsize=(8,8))
        # origin image
        origin_image = image_float_to_uint8(image.copy())
        pltdrawer.add_subplot(origin_image, "origin image")

        # result image
        result_image = image_float_to_uint8(image.copy())
        for human in humans:
            result_image = human.draw_human(result_image)
        pltdrawer.add_subplot(result_image, "result image")

        # save figure
        pltdrawer.savefig(f"{self.save_dir}/{name}.png")

    def visualize(self, image_batch, predict_x, mask_batch=None, humans_list=None, name="vis"):
        raise NotImplementedError("abstract class BasicVisualizer function: visualize not implemented!")
    
    def visualize_compare(self, image_batch, predict_x, target_x, mask_batch=None, humans_list=None, name="vis"):
        raise NotImplementedError("abstract class BasicVisualizer function: visualize_compare not implemented!")

class PltDrawer:
    def __init__(self, draw_row, draw_col, figsize=(8,8), dpi=300):
        self.draw_row = draw_row
        self.draw_col = draw_col
        self.figsize = figsize
        self.dpi = dpi
        self.plot_images=[]
        self.plot_titles=[]
        self.color_bars=[]

    def add_subplot(self,plot_image, plot_title, color_bar=False):
        self.plot_images.append(plot_image)
        self.plot_titles.append(plot_title)
        self.color_bars.append(color_bar)
    
    def draw_plots(self):
        fig = plt.figure(figsize=self.figsize)
        for draw_idx,(image, title, color_bar) in enumerate(zip(self.plot_images,self.plot_titles,self.color_bars)):
            a = fig.add_subplot(self.draw_row, self.draw_col, draw_idx+1)
            a.set_title(title)
            plt.imshow(image)
            if(color_bar):
                plt.colorbar()
    
    def savefig(self,save_path):
        self.draw_plots()
        plt.savefig(save_path,dpi=self.dpi)
        plt.close()

class ImageProcessor:
    def __init__(self, input_h, input_w):
        self.input_h = input_h
        self.input_w = input_w
    
    def read_image_rgb_float(self, image_path):
        # return an image with rgb channel order and float value within [0,1] 
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.clip(image.astype(np.float32)/255.0,0.0,1.0).astype(np.float32)
        return image

    def write_image_rgb_float(self, image, image_path):
        # write an image which has rgb channel order and float value within [0,1]
        image = np.clip(image*255.0, 0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.imwrite(image_path, image)
    
    def image_pad_and_scale(self, image):
        # pad and scale image to input_h and input_w
        # output scaled image with pad
        image_h ,image_w, _ =image.shape
        scale = min(self.input_h/image_h, self.input_w/image_w)
        scale_h, scale_w = int(scale*image_h), int(scale*image_w)
        scaled_image = cv2.resize(image, (scale_w,scale_h), interpolation=cv2.INTER_CUBIC)
        pad_image, pad = pad_image_shape(scaled_image, shape=[self.input_h, self.input_w])
        pad_image = pad_image.astype(np.float32)
        return pad_image, scale, pad
        
        