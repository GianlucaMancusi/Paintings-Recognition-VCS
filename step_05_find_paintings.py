import cv2
import numpy as np
import math
import random
from step_04_connected_components import color_contours

def couldBePainting(img: np.array, bounder, contour, width, height, area_percentage):
    bounder_width, bounder_height = bounder[2], bounder[3]
    bounder_area = bounder_width * bounder_height
    # Check that the rect is smaller than the entire image and bigger than a certain size
    if bounder_area < img.shape[0]*img.shape[1]*0.9 and bounder_width > width and bounder_height > height:
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
    from data_test.standard_samples import RANDOM_PAINTING, TEST_PAINTINGS
    from pipeline import Pipeline

    filename = TEST_PAINTINGS[2]
    step = 5

    img = cv2.imread(filename)
    pipeline = Pipeline()
    pipeline.set_default(step)
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)
    pipeline.debug_history().show()
    cv2.imwrite(f'data_test/{step:02d}.jpg', out)
