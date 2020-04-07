import cv2 as cv
import numpy as np
from step_07_median_filter import apply_median_filter

"""
    Applies Canny's edge detector
    
    Parameters
    ----------
    img
        the image

    Returns
    -------
    img
        the image's edges
"""
def apply_edge_detection(img):
    return cv.Canny(img, 50, 100)


if __name__ == "__main__":
    img = cv.imread("data_test/median_filter_sample.png")
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    filtered = apply_median_filter(img)
    edges = apply_edge_detection(filtered)
    cv.imshow("Original", img)
    cv.imshow("Filtered", edges)

    cv.waitKey(0)
