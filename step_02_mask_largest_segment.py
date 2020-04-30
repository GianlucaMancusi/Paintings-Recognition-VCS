import cv2
import numpy as np
import math
import random


# FLOORFILL
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill


def mask_largest_segment(input: np.array, color_difference=2, delta=32, scale_percent=1.0, x_samples=64, debug=False):
    """
    The largest segment will be white and the rest is black

    Useful to return a version of the image where the wall 
    is white and the rest of the image is black.

    see more: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill
    ----------
    img : np.array
        image where to find the largest element

    color_difference : int
        The distance from colors to permit.

    x_samples : int
        numer of samples that will be tested orizontally in the image
    """
    im = input.copy()

    h = im.shape[0]
    w = im.shape[1]

    if scale_percent != 1.0:
        height = int(h * scale_percent)
        width = int(w * scale_percent)
        # resize image
        im = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
    
    # in that way for smaller images the stride will be lower
    stride = int(w / x_samples)

    mask = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=np.uint8)
    wallColor = np.zeros((1, 1, 1))
    largest_segment = 0
    for y in range(0, im.shape[0], stride):
        for x in range(0, im.shape[1], stride):
            if mask[y+1, x+1] == 0:
                point_colour = (int(im[y, x, 0]), int(im[y, x, 1]), int(im[y, x, 2]))
                # Fills a connected component with the given color.
                # loDiff – Maximal lower brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # upDiff – Maximal upper brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # flags=4 means that only the four nearest neighbor pixels (those that share an edge) are considered.
                #       8 connectivity value means that the eight nearest neighbor pixels (those that share a corner) will be considered
                rect = cv2.floodFill(
                    im, mask, (x, y), point_colour, loDiff=color_difference, upDiff=color_difference, flags=4)
                segment_size = rect[3][2]*rect[3][3]
                if segment_size > largest_segment:
                    largest_segment = segment_size
                    wallColor = point_colour


    # checks if our image pixel values are the same of the wallColor's pixel values.
    lowerBound = tuple([max(x - delta, 0) for x in wallColor])
    upperBound = tuple([min(x + delta, 255) for x in wallColor])
    wallmask = cv2.inRange(im, lowerBound, upperBound)

    wallmask = cv2.resize(wallmask, (w, h), interpolation=cv2.INTER_AREA)  # strideeedup

    if debug:
        return wallmask, wallmask
    else:
        return wallmask


if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING
    from pipeline import Pipeline
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(2)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()
