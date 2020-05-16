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
def _clean_frames_noise(img):
    kernel = np.ones((23, 23), np.uint8)
    # eroded = cv2.erode(img, kernel, iterations=5)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
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

if __name__ == "__main__":
    from pipeline import Pipeline
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(6)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()
