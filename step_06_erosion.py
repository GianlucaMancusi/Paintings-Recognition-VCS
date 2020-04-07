import cv2 as cv
import numpy as np

"""
    Cleans the noise around paintings' frames though erosion
    
    Parameters
    ----------
    img
        the image to be cleaned

    Returns
    -------
    img
        the cleaned image
"""
def clean_frames_noise(img):
    kernel = np.ones((23, 23), np.uint8)
    eroded = cv.erode(img, kernel, iterations=5)
    
    return eroded


if __name__ == "__main__":
    img = cv.imread('data_test/erosion-of-frame-components.png',)
    
    eroded = clean_frames_noise(img)
    cv.imshow("Original", img)
    cv.imshow("Eroded", eroded)
    cv.waitKey(0)
