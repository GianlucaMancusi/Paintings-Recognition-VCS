import cv2
import numpy as np
import math

def highlight_paintings(input, source, pad=0, debug=False):
    """
    Given an image and the four corners of the picture it returns a copy of the
    image with the picture's contours drawn.
    
    Parameters
    ----------
    img : np.array
        image where the contours will be drawn
    corners : list
        list of corners [x, y]

    Returns
    -------
    image
        image in RGB where there is drawn the cotnours are drawn
    """
    corners_list = input
    polyImg = source.copy()

    for corners in corners_list:
        corners = [(x - pad, y - pad) for x, y in corners]

        pts = np.array(corners, np.int32)
        pts = cv2.convexHull(pts)
        pts = pts.reshape((-1,1,2))

        cv2.polylines(polyImg, [pts], True, (231, 76, 60), thickness=3)
    if debug:
        return polyImg, polyImg
    else:
        return polyImg

if __name__ == '__main__':
    from pipeline import Pipeline, Function
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline(default=True)
    pipeline.append(Function(highlight_paintings, source=img, pad=100))
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()