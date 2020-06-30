import cv2
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
def _clean_frames_noise(img, k_size=23, iterations=1):
    kernel = np.ones((k_size, k_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    return opening

def clean_frames_noise(input, debug=False):
    opening = _clean_frames_noise(input)    
    if debug:
        return opening, opening
    else:
        return opening

def _mask_from_contour(img, contour):
    canvas = np.zeros_like(img)
    cv2.fillPoly(canvas, pts=[contour], color=(255, 255, 255))
    return canvas

def mask_from_contour(input, debug=False):
    img, contour = input
    canvas = _mask_from_contour(img, contour)
    if debug:
        return canvas, canvas
    else:
        return canvas


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


def _apply_edge_detection(img, t1=50, t2=100):
    result = cv2.Canny(img, t1, t2)
    return result


def apply_edge_detection(input, debug=False):
    result = _apply_edge_detection(input)
    if debug:
        kernel = np.ones((5, 5), np.uint8)
        debug_img = cv2.dilate(result, kernel)
        return result, debug_img
    else:
        return result

if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING, TEST_PAINTINGS
    from pipeline import Pipeline

    filename = TEST_PAINTINGS[2]
    step = 6

    img = cv2.imread(filename)
    pipeline = Pipeline()
    pipeline.set_default(step)
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)
    pipeline.debug_history().show()
    cv2.imwrite(f'data_test/{step:02d}.jpg', out)
