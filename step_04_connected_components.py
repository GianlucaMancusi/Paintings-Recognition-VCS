import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from step_01_mean_shift_seg import mean_shift_segmentation
from step_02_mask_largest_segment import mask_largest_segment
from step_03_dilate_invert import erode_dilate, invert

import random


def findContours(img: np.array):
    """
    Dilates an image by using a specific structuring element
    The function retrieves contours from the binary image using the algorithm [Suzuki85].

    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html?highlight=dilate#dilate
    ----------
    img : np.array
        image where to apply the dilatation
    """
    # CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours.
    # CV_CHAIN_APPROX_NONE stores absolutely all the contour points.
    # That is, any 2 subsequent points (x1,y1) and (x2,y2) of the contour will be either horizontal, vertical or diagonal neighbors,
    #  that is, max(abs(x1-x2),abs(y2-y1))==1.
    contours, hierarchy = cv2.findContours(
        img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours


def couldBePainting(img: np.array, bounder, contour, width, height, area_percentage):
    bounder_area = bounder[2]*bounder[3]
    # Check that the rect is smaller than the entire image and bigger than a certain size
    if bounder_area < img.shape[0]*img.shape[1] and bounder_area > width*height:
        # Extra to remove floors when programming
        if cv2.contourArea(contour) > bounder_area*.6:
            return True
    return False


def findPossibleContours(img: np.array, contours, min_width=150, min_height=150, min_area_percentage=.6):
    painting_contours = []
    for contour in contours:
        bounder = cv2.boundingRect(contour)
        if couldBePainting(img, bounder, contour, min_width, min_height, min_area_percentage):
            painting_contours.append(contour)
    return painting_contours


if __name__ == "__main__":
    rgbImage = cv2.imread('data_test/gallery_0.jpg')
    meanshiftseg = mean_shift_segmentation(rgbImage)
    mask_largest = mask_largest_segment(meanshiftseg)
    final_mask = erode_dilate(mask_largest)
    inversion = invert(final_mask)

    contours = findContours(inversion)
    painting_contours = findPossibleContours(inversion, contours)
    for painting_contour in painting_contours:
        color1 = (list(np.random.choice(range(256), size=3)))
        color = [int(color1[0]), int(color1[1]), int(color1[2])]
        cv2.fillPoly(inversion, pts=painting_contour, color=color)

    f, axarr = plt.subplots(1, 2)
    rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)
    inversion = cv2.cvtColor(inversion, cv2.COLOR_BGR2RGB)
    axarr[0].imshow(rgbImage)
    axarr[1].imshow(inversion)
    plt.show()
    pass
