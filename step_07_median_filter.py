import cv2
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
def _apply_median_filter(img, strength=15):
    result = cv2.medianBlur(img, strength)
    return result

def apply_median_filter(input, debug=False):
    result = _apply_median_filter(input)
    if debug:
        return result, result
    else:
        return result


if __name__ == "__main__":
    from pipeline import Pipeline
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(7)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()
