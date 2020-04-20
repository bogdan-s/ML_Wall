"""
Refine the masks and create separate edges mask
"""


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import concurrent.futures

old_masks_path = 'D:/Python/DataSets/ADE20K_Filtered/Train/Masks/0/'
noGaps_dest = 'D:/Python/DataSets/ADE20K_Filtered/Train/New_Masks/0/'
edges_dest = 'D:/Python/DataSets/ADE20K_Filtered/Train/Edges/0/'

rgb_imgs = 'D:/Python/DataSets/ADE20K_Filtered/Train/Images/0/'

def write_new_masks(img, NoGaps, edges, img_name):
    # plot_images(img, NoGaps, edges)
    # print("dsfasdfasd")
    cv2.imwrite(noGaps_dest + img_name + '_seg.png', NoGaps)
    cv2.imwrite(edges_dest + img_name + '_edg.png', edges)
    print(img_name)

def create_new_masks(img_path, img_name):
    img = cv2.imread(img_path,0)
    # remove small gaps in masks
    kernel = np.ones((3,3),np.uint8)
    NoGaps = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    NoGaps = cv2.morphologyEx(NoGaps, cv2.MORPH_OPEN, kernel)

    # edge detection kernel
    kernel = np.array((
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]), dtype="float32")/25


    edges = cv2.filter2D(NoGaps,-1, kernel)
    h, w = edges.shape
    for i in range(h):
        for j in range(w):
            if edges[i, j] > 0:
                edges[i, j] = 255 #make edges white (no intermediate values)
    # print("writing")
    write_new_masks(img, NoGaps, edges, img_name)


def plot_images(img, NoGaps, edges):
    plt.subplot(131),plt.imshow(img, cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow(NoGaps, cmap = 'gray')
    plt.title('NoGaps Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow(edges, cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()

mask_names = []
old_mask_paths = []
for root, dirs, files in os.walk(rgb_imgs):
    for img in files:
        mask_names.append(img[:-4])
        old_mask_paths.append(os.path.join(old_masks_path, img[:-4] + "_seg.png"))
        # break
if __name__ == '__main__':
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.map(create_new_masks, old_mask_paths, mask_names)