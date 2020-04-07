import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from step_01_mean_shift_seg import mean_shift_segmentation
import random


def floorFill(img: np.array, color_difference):
    mask = np.zeros(img)


# CIAO
# Stavo studiando da qui
# https://github.com/nating/recognizing-paintings/blob/9ca9ba0720f71d451ffb706da950631d780acc0b/src/assignment-2/main.cpp#L135
# linea 135

# ho però implementato questa https://github.com/nating/recognizing-paintings/blob/9ca9ba0720f71d451ffb706da950631d780acc0b/src/assignment-2/vision-techniques.h#L60
# linea 60 (per provare)
# non va a prendere randomicamente i colori ma prende il colore del pixel analizzato

# LA DOCUMANTAZIONE DI FLOORFILL
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill

def mask_largest_segment(img: np.array, color_difference):
    """
    The largest segment will be white and the rest is black

    Useful to return a version of the image where the wall 
    is white and the rest of the image is black.

    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill
    ----------
    img : np.array
        image where to find the largest element

    color_difference : int
        The distance from colors to permit.
    """
    im = img.copy()
    mask = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=np.uint8)
    wallColor = np.zeros((1, 1, 1))
    largest_segment = 0
    for y in range(im.shape[0]):
        for x in range(im.shape[1]):
            if mask[y+1, x+1] == 0:
                point = (x, y)
                point_colour = (int(im[y,x,0]),int(im[y,x,1]),int(im[y,x,2]))
                # Fills a connected component with the given color.
                # loDiff – Maximal lower brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # upDiff – Maximal upper brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # flags=4 means that only the four nearest neighbor pixels (those that share an edge) are considered.
                #       8 connectivity value means that the eight nearest neighbor pixels (those that share a corner) will be considered
                rect = cv2.floodFill(
                    im, mask, (x, y), point_colour, loDiff=color_difference, upDiff=color_difference, flags=4)
                segment_size = rect[3][2]*rect[3][3]
                if segment_size > largest_segment:
                    largest_segment = segment_size
                    wallColor = point_colour

    # checks if our image pixel values are the same of the wallColor's pixel values.
    delta = 24
    lowerBound = tuple([x - delta for x in wallColor])
    upperBound = tuple([x + delta for x in wallColor])
    wallmask = cv2.inRange(im, lowerBound, upperBound)
    return wallmask


if __name__ == "__main__":
    rgbImage = cv2.imread('data_test/gallery_0.jpg')
    meanshiftseg = mean_shift_segmentation(
        rgbImage, spatial_radius=7, color_radius=30, maximum_pyramid_level=1)
    pre = meanshiftseg.copy()
    final_mask = mask_largest_segment(meanshiftseg, 2)
    f, axarr = plt.subplots(2, 2)
    rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)
    pre = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
    meanshiftseg = cv2.cvtColor(meanshiftseg, cv2.COLOR_BGR2RGB)
    final_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB)
    axarr[0,0].imshow(rgbImage)
    axarr[0,1].imshow(pre)
    axarr[1,0].imshow(meanshiftseg)
    axarr[1,1].imshow(final_mask)
    plt.show()
    pass
