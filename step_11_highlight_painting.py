import cv2
import numpy as np
import math

def highlight_paintings(input, source, pad=0, debug=False):
    polyImg = _highlight_paintings(input, source, pad)
    if debug:
        return polyImg, polyImg
    else:
        return polyImg

def _highlight_paintings(corners_list, source, pad=0, debug=False):
    """
    Given an image and the four corners of the picture it returns a copy of the
    image with the picture's contours drawn.
    
    Parameters
    ----------
    source : np.array
        image where the contours will be drawn
    corners_list : list
        list of corners [x, y]

    Returns
    -------
    image
        image in RGB where there is drawn the cotnours are drawn
    """
    polyImg = source.copy()
    corners = []
    for corner in corners_list:
        try:
            bb = [(x - pad, y - pad) for x, y in corner]
            corners.append(bb)
        except Exception:
            continue

    pts = np.array(corners, np.int32)
    for bb in pts:
        convexHull = cv2.convexHull(bb)
        convexHull = convexHull.reshape((-1,1,2))
        cv2.polylines(polyImg, [convexHull], True, (231, 76, 60), thickness=3)
    
    return polyImg

def _draw_all_contours(contours, img):
    res = img.copy()
    cv2.polylines(res, contours, True, (231, 76, 60), thickness=3)
    return res

if __name__ == '__main__':
    from pipeline import Pipeline, Function
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline(default=True)
    pipeline.append(Function(highlight_paintings, source=img, pad=100))
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()