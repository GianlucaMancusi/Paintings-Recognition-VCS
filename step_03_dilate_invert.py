import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from step_01_mean_shift_seg import mean_shift_segmentation
from step_02_mask_largest_segment import mask_largest_segment

import random


def erode_dilate(img: np.array, size=5, erode=True):
    """
    Dilates an image by using a specific structuring element

    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#dilate
    ----------
    img : np.array
        image where to apply the dilatation
    """
    kernel = np.ones((size, size), np.uint8)
    if erode: 
        img = cv2.erode(img, kernel)
    img = cv2.dilate(img, kernel)
    return img

def invert(img: np.array):
    """
    White becomes Black and viceversa
    ----------
    img : np.array
        image where to apply the inversion
    """
    return 255-img

def erode_dilate_invert(img:np.array, size=5, erode=True):
    inversion = invert(erode_dilate(img,size, erode))
    return inversion

if __name__ == "__main__":
    rgbImage = cv2.imread('data_test/gallery_0.jpg')
    meanshiftseg = mean_shift_segmentation(rgbImage)
    mask_largest = mask_largest_segment(meanshiftseg)
    final_mask = erode_dilate(mask_largest)
    inversion = invert(final_mask)
    f, axarr = plt.subplots(1,3)
    mask_largest = cv2.cvtColor(mask_largest, cv2.COLOR_BGR2RGB)
    final_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB)
    inversion = cv2.cvtColor(inversion, cv2.COLOR_BGR2RGB)
    axarr[0].imshow(mask_largest)
    axarr[1].imshow(final_mask)
    axarr[2].imshow(inversion)
    plt.show()
    pass
