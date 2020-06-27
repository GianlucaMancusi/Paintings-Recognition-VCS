import cv2
import numpy as np
import math
import random

def color_contours(img, contours):
    canvas = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # canvas = np.zeros_like(img)
    for contour in contours:
        color = np.random.randint(256, size=3).tolist()
        cv2.fillPoly(canvas, pts=[contour], color=color)
    return canvas

def find_contours(input: np.array, debug=False):
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
    img = input
    contours = _find_contours(img)
    if debug:
        canvas = color_contours(img.copy(), contours)
        return (img, contours), canvas
    else:
        return (img, contours)

def _find_contours(img: np.array):
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
    contours, hierarchy = cv2.findContours(img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE) # cv2.CHAIN_APPROX_SIMPLE to save memory
    return contours

if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING, TEST_PAINTINGS
    from pipeline import Pipeline

    filename = TEST_PAINTINGS[2]
    step = 4

    img = cv2.imread(filename)
    pipeline = Pipeline()
    pipeline.set_default(step)
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)
    pipeline.debug_history().show()
    cv2.imwrite(f'data_test/{step:02d}.jpg', out)
