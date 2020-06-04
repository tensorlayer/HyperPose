import cv2
import numpy as np
import _pickle as cpickle
import matplotlib.pyplot as plt

def visualize(vis_dir,vis_num,dataset,parts,colors):
    for vis_id,(img_file,annos) in enumerate(dataset,start=1):
        if(vis_id>=vis_num):
            break
        image=cv2.cvtColor(cv2.imread(img_file.numpy().decode("utf-8"),cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        annos=cpickle.loads(annos.numpy())
        ori_img=image
        vis_img=image.copy()
        kpts_list=annos[0]
        print(f"visualizing image:{vis_id} with {len(kpts_list)} humans...")
        for kpts in kpts_list:
            x,y,v=kpts[0::3],kpts[1::3],kpts[2::3]
            for part_idx in range(0,len(parts)):
                color=colors[part_idx]
                vis_img=cv2.circle(vis_img,(int(x[part_idx]),int(y[part_idx])),radius=6,color=color,thickness=-1)
        fig=plt.figure(figsize=(8,8))
        a=fig.add_subplot(1,2,1)
        a.set_title("original image")
        plt.imshow(ori_img)
        a=fig.add_subplot(1,2,2)
        a.set_title("visualized image")
        plt.imshow(vis_img)
        plt.savefig(f"{vis_dir}/{vis_id}_vis_coco.png")
        plt.close('all')