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
    from data_test.standard_samples import RANDOM_PAINTING, TEST_PAINTINGS
    from pipeline import Pipeline, Function

    filename = TEST_PAINTINGS[0]

    img = cv2.imread(filename)
    pipeline = Pipeline(default=True)
    pipeline.append(Function(highlight_paintings, source=img, pad=100))
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)
    for step, out in enumerate([img, ] + pipeline.debug_out_list):
        cv2.imwrite(f'data_test/{step:02d}.jpg', out)