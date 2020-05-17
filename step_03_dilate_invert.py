import cv2
import numpy as np
import math


def erode_dilate(input: np.array, size=5, erode=True, debug=False):
    img = _erode_dilate(input, size, erode)
    if debug:
        return img, img
    else:
        return img

def _erode_dilate(img: np.array, size=5, erode=True):
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

def invert(input: np.array, debug=False):
    result = _invert(input)
    if debug:
        return result, result
    else:
        return result
        
def _invert(img: np.array):
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

def _add_padding(img, pad=100, color=[0, 0, 0]):
    result = cv2.copyMakeBorder(
            img,
            top=pad,
            bottom=pad,
            left=pad,
            right=pad,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    return result

def add_padding(input, pad=100, color=[0, 0, 0], debug=False):
    result = _add_padding(input, pad, color)
    if debug:
        return result, result
    else:
        return result

if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING
    from pipeline import Pipeline
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(3)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()
