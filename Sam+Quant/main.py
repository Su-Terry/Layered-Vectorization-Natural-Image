from cv2 import kmeans, TERM_CRITERIA_MAX_ITER, TERM_CRITERIA_EPS, KMEANS_RANDOM_CENTERS, imread, cvtColor, COLOR_BGR2RGB
from numpy import float32, uint8, unique
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, imshow
from matplotlib.pyplot import savefig
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import os

def quant(img, k, name):
    img_RGB = cvtColor(img, COLOR_BGR2RGB)
    img_data = img_RGB.reshape(-1, 3)

    print(len(unique(img_data, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')
    criteria = (TERM_CRITERIA_MAX_ITER + TERM_CRITERIA_EPS, 10, 1.0)
    compactness, labels, centers = kmeans(data=img_data.astype(float32), K=k, bestLabels=None, criteria=criteria, attempts=10, flags=KMEANS_RANDOM_CENTERS)
    colours = centers[labels].reshape(-1, 3)
    print(len(unique(colours, axis=0)), 'unique RGB values out of', img_data.shape[0], 'pixels')
    img_colours = colours.reshape(img_RGB.shape)
    img_colours = cv2.blur(img_colours, (5, 5))  
    imshow(img_colours.astype(uint8))
    show()
    plt.axis('off')
    savefig('out_quan/'+name+'_result.png', bbox_inches='tight', pad_inches=0)
    return img_colours


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [1.00]])
        img[m] = color_mask

    ax.imshow(img)

    return img
    

def SAM(name):
    # image = cv2.imread('images/dog_result.png')
    image = cv2.imread('out_quan/' + name )
    height, width, _ = image.shape
    black_image = np.zeros((height, width, 3), dtype=np.uint8)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    masks = mask_generator.generate(image)
    image2 = show_anns(masks)

    print(len(masks))
    print(masks[0].keys())

    plt.figure(figsize=(20, 20))
    plt.imshow(black_image)  
    show_anns(masks)
    plt.axis('off')
    plt.savefig('output/' + name + '_result1.png', bbox_inches='tight', pad_inches=0)
    plt.show()

    mask_generator_2 = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side = 64, #32  #up detail 
        pred_iou_thresh = 0.26, #0.86     #orange 26
        stability_score_thresh = 0.88, #0.92  #orange 88
        crop_n_layers = 1, 
        crop_n_points_downscale_factor=2,  #2
        min_mask_region_area= 20,  # Requires open-cv to run post-processing #20
    )

    masks2 = mask_generator_2.generate(image)
    image3 = show_anns(masks2)
    len(masks2)

    plt.figure(figsize=(20, 20))
    plt.imshow(black_image)  # 創建新的 black_image
    show_anns(masks2)
    plt.axis('off')
    plt.savefig('output/' + name + '_result2.png', bbox_inches='tight', pad_inches=0)
    plt.show()


#---------------------MAIN--------------------------#
folder_path = 'simple_data/' 
if os.path.exists(folder_path):
    print("----------------------Quan---------------------------")
    files = os.listdir(folder_path)
    image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    for name in image_files:
        img, img_quan, img_out = [], [], []
        name_real = os.path.splitext(name)[0]
        k=8
        print(name_real)
        img = imread('simple_data/'+name)
        img_quan = quant(img , k, name_real)  
    files2 = os.listdir('out_quan/')
    image_files2 = [file2 for file2 in files2 if file2.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    print("----------------------SAM---------------------------")
    for name2 in image_files2:
        print(name2)
        img_out = SAM(name2)
        

else:
    print("指定的資料夾不存在")
