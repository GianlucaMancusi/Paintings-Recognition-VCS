import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from step_01_mean_shift_seg import mean_shift_segmentation
from step_02_mask_largest_segment import mask_largest_segment
from step_03_dilate_invert import erode_dilate, invert
from step_04_connected_components import findContours
import random

def couldBePainting(img: np.array, bounder, contour, width, height, area_percentage):
    bounder_area = bounder[2]*bounder[3]
    # Check that the rect is smaller than the entire image and bigger than a certain size
    if bounder_area < img.shape[0]*img.shape[1] and bounder[2] >= width and bounder[3] >= height:
        # Extra to remove floors when programming
        if cv2.contourArea(contour) > bounder_area * area_percentage:
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
    #dataset\\photos\000\\VIRB0399\\000090.jpg
    rgbImage = cv2.imread('dataset\\photos\\000\\VIRB0399\\000090.jpg')
    meanshiftseg = mean_shift_segmentation(rgbImage)
    mask_largest = mask_largest_segment(meanshiftseg)
    final_mask = erode_dilate(mask_largest)
    inversion = invert(final_mask)

    contours = findContours(inversion)
    painting_contours = findPossibleContours(inversion, contours)
    inversion = cv2.cvtColor(inversion, cv2.COLOR_GRAY2BGR)
    for painting_contour in painting_contours:
        color1 = (list(np.random.choice(range(256), size=3)))
        color = [int(color1[0]), int(color1[1]), int(color1[2])]
        cv2.fillPoly(inversion, pts=[painting_contour], color=color)

    f, axarr = plt.subplots(1, 2)
    rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)
    inversion = cv2.cvtColor(inversion, cv2.COLOR_BGR2RGB)
    axarr[0].imshow(rgbImage)
    axarr[1].imshow(inversion)
    plt.show()
    pass
