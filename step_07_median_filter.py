import cv2 as cv
import numpy as np


"""
    Applies the proper median filter for smoothing the frame's sides
    
    Parameters
    ----------
    img
        the image

    Returns
    -------
    img
        the smoothed image
"""
def apply_median_filter(img):
    return cv.medianBlur(img, 31)


if __name__ == "__main__":
    img = cv.imread("data_test/median_filter_sample.png")
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    filtered = apply_median_filter(img)
    cv.imshow("Original", img)
    cv.imshow("Filtered", filtered)

    cv.waitKey(0)
