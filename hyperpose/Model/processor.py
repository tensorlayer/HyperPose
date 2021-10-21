import matplotlib.pyplot as plt
# Basic class of processors to be inherit
class BasicPreProcessor:
    def __init__(self, parts, limbs, hin, win, hout, wout, colors=None, data_format="channels_first"):
        self.parts = parts
        self.limbs = limbs
        self.hin, self.win = hin, win
        self.hout, self.wout = hout, wout
        self.colors = colors
        self.data_format = data_format
    
    def process(self, annos, mask, bbxs):
        raise NotImplementedError("abstract class BasicPreProcessor function: process not implemented!")

class BasicPostProcessor:
    def __init__(self, parts, limbs, colors=None, data_format="channels_first"):
        self.parts = parts
        self.limbs = limbs
        self.colors = colors
        self.data_format = data_format
    
    def process(self, predict_x):
        raise NotImplementedError("abstract class BasicPostProcessor function: process not implemented!")

class BasicVisualizer:
    def __init__(self, save_dir="./save_dir"):
        self.save_dir = save_dir
    
    def set_save_dir(self, save_dir):
        self.save_dir = save_dir
    
    def visualize_result(self, image, humans, save_path):
        pltdrawer = PltDrawer(draw_row=1, draw_col=2, figsize=(8,8))
        # origin image
        pltdrawer.add_subplot(image, "origin image")

        # result image
        result_image = image.copy()
        for human in humans:
            result_image = human.draw(result_image)
        pltdrawer.add_subplot(result_image, "result image")

        # save figure
        pltdrawer.savefig(save_path)

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
        plt.savefig(save_path,dpi=self.dpi)
        plt.close()
