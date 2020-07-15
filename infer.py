import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hyperpose import Config,Model,Dataset
from hyperpose.Dataset import imread_rgb_float,imwrite_rgb_float
Config.set_model_name("openpose")
Config.set_model_type(Config.MODEL.Openpose)
config=Config.get_config()

#get and load model
model=Model.get_model(config)
weight_path=f"{config.model.model_dir}/newest_model.npz"
model.load_weights(weight_path)

#infer on single image
ori_image=cv2.cvtColor(cv2.imread("./sample.jpg"),cv2.COLOR_BGR2RGB)
input_image=ori_image.astype(np.float32)/255.0
if(model.data_format=="channels_first"):
    input_image=np.transpose(input_image,[2,0,1])

img_c,img_h,img_w=input_image.shape
conf_map,paf_map=model.infer(input_image[np.newaxis,:,:,:])

#get visualize function, which is able to get visualized part and limb heatmap image from inferred heatmaps
visualize=Model.get_visualize(Config.MODEL.Openpose)
vis_parts_heatmap,vis_limbs_heatmap=visualize(input_image,conf_map[0],paf_map[0],save_tofile=False,)

#get postprocess function, which is able to get humans that contains assembled detected parts from inferred heatmaps
postprocess=Model.get_postprocess(Config.MODEL.Openpose)
humans=postprocess(conf_map[0],paf_map[0],img_h,img_w,model.parts,model.limbs,model.data_format,model.colors)
#draw all detected skeletons
output_img=ori_image.copy()
for human in humans:
    output_img=human.draw_human(output_img)

#if you want to visualize all the images in one plot:
#show image,part heatmap,limb heatmap and detected image
#here we use 'transpose' because our data_format is 'channels_first'
fig=plt.figure(figsize=(8,8))
#origin image
origin_fig=fig.add_subplot(2,2,1)
origin_fig.set_title("origin image")
origin_fig.imshow(ori_image)
#parts heatmap
parts_fig=fig.add_subplot(2,2,2)
parts_fig.set_title("parts heatmap")
parts_fig.imshow(vis_parts_heatmap)
#limbs heatmap
limbs_fig=fig.add_subplot(2,2,3)
limbs_fig.set_title("limbs heatmap")
limbs_fig.imshow(vis_limbs_heatmap)
#detected results
result_fig=fig.add_subplot(2,2,4)
result_fig.set_title("detect result")
result_fig.imshow(output_img)
#save fig
plt.savefig("./sample_custome_infer.png")
plt.close()
