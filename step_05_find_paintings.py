import cv2
import numpy as np
import math
import random
from step_04_connected_components import color_contours

def couldBePainting(img: np.array, bounder, contour, width, height, area_percentage):
    bounder_area = bounder[2]*bounder[3]
    # Check that the rect is smaller than the entire image and bigger than a certain size
    if bounder_area < img.shape[0]*img.shape[1] and cv2.contourArea(contour) > width*height:
        # Extra to remove floors when programming
        if cv2.contourArea(contour) > bounder_area * area_percentage:
            return True
    return False


def find_possible_contours(input, min_width=50, min_height=50, min_area_percentage=.6, debug=False):
    img, contours = input
    painting_contours = _find_possible_contours(img, contours, min_width, min_height, min_area_percentage)
    result = [(img, contour) for contour in painting_contours]
    if debug:
        canvas = color_contours(img.copy(), painting_contours)
        return result, canvas
    else:
        return result

def _find_possible_contours(img, contours, min_width=50, min_height=50, min_area_percentage=.6):
    painting_contours = []
    for contour in contours:
        bounder = cv2.boundingRect(contour)
        if couldBePainting(img, bounder, contour, min_width, min_height, min_area_percentage):
            painting_contours.append(contour)
    return painting_contours


if __name__ == "__main__":
    from pipeline import Pipeline
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(5)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()
