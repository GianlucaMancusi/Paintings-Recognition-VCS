import cv2
import numpy as np
import math


def mask(input, source, pad=0, debug=False):
    """
    Given an B&W image it returns an image with
    the bounding box drawn.
    
    Parameters
    ----------
    img : np.array
        in black and white

    Returns
    -------
    image
        image in binary where there is drawn the mask
    """
    contours = [ contour for img, contour in input ]
    img = source.copy()
    for contour in contours:
        rect = cv2.boundingRect(contour)
        x,y,w,h = rect
        x -= pad
        y -= pad
        img = cv2.rectangle(img, (x, y), (x+w, y+h), [0, 255, 0], thickness=3)
    if debug:
        return img, img
    else:
        return img


if __name__ == '__main__':
    from pipeline import Pipeline, Function
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(5)
    pipeline.append(Function(mask, source=img, pad=100))
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()