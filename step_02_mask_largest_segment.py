import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from step_01_mean_shift_seg import mean_shift_segmentation
import random


# FLOORFILL
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill

def mask_largest_segment(img: np.array, color_difference=2):
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
    lowerBound = tuple([max(x - delta,0) for x in wallColor])
    upperBound = tuple([min(x + delta,255) for x in wallColor])
    wallmask = cv2.inRange(im, lowerBound, upperBound)
    return wallmask


if __name__ == "__main__":
    rgbImage = cv2.imread('data_test/gallery_0.jpg')
    meanshiftseg = mean_shift_segmentation(rgbImage)
    final_mask = mask_largest_segment(meanshiftseg, 2)
    f, axarr = plt.subplots(1,2)
    meanshiftseg = cv2.cvtColor(meanshiftseg, cv2.COLOR_BGR2RGB)
    final_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2RGB)
    axarr[0].imshow(meanshiftseg)
    axarr[1].imshow(final_mask)
    plt.show()
    pass
