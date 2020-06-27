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
    from data_test.standard_samples import RANDOM_PAINTING, TEST_PAINTINGS
    from pipeline import Pipeline

    filename = TEST_PAINTINGS[2]
    step = 7

    img = cv2.imread(filename)
    pipeline = Pipeline()
    pipeline.set_default(step)
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)
    pipeline.debug_history().show()
    cv2.imwrite(f'data_test/{step:02d}.jpg', out)
