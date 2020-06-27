import cv2
import numpy as np
import math
import random


# FLOORFILL
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=floodfill


def mask_largest_segment(input: np.array, debug=False, **kwargs):
    wallmask = _mask_largest_segment(input, **kwargs)

    if debug:
        return wallmask, wallmask
    else:
        return wallmask

def _mask_largest_segment(im: np.array, color_difference=2, scale_percent=1.0, x_samples=2, no_skip_white=False):
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
    im = im.copy()
    
    h = im.shape[0]
    w = im.shape[1]
    color_difference = (color_difference,) * 3
    
    # in that way for smaller images the stride will be lower
    stride = int(w / x_samples)

    mask = np.zeros((im.shape[0]+2, im.shape[1]+2), dtype=np.uint8)
    wallmask = mask[1:-1,1:-1].copy()
    largest_segment = 0
    for y in range(0, im.shape[0], stride):
        for x in range(0, im.shape[1], stride):
            if mask[y+1, x+1] == 0 or no_skip_white:
                mask[:] = 0
                # Fills a connected component with the given color.
                # loDiff – Maximal lower brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # upDiff – Maximal upper brightness/color difference between the currently observed pixel and one of its neighbors belonging to the component, or a seed pixel being added to the component.
                # flags=4 means that only the four nearest neighbor pixels (those that share an edge) are considered.
                #       8 connectivity value means that the eight nearest neighbor pixels (those that share a corner) will be considered
                rect = cv2.floodFill(
                    im.copy(),
                    mask,
                    (x, y),
                    0,
                    color_difference,
                    color_difference,
                    flags=4 | ( 255 << 8 ),
                    )
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                segment_size = mask.sum()
                if segment_size > largest_segment:
                    largest_segment = segment_size
                    wallmask = mask[1:-1,1:-1].copy()
                    # cv2.imshow('rect[2]', mask)
                    # cv2.waitKey(0)
    wallmask = wallmask.astype(np.int64) + ((im.sum(2) == 0).astype(np.int64) * 255)
    wallmask = np.clip(wallmask, 0, 255)
    return wallmask.astype(np.uint8)


if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING, TEST_PAINTINGS
    from pipeline import Pipeline

    filename = TEST_PAINTINGS[2]
    step = 2

    img = cv2.imread(filename)
    pipeline = Pipeline()
    pipeline.set_default(step)
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)
    pipeline.debug_history().show()
    cv2.imwrite(f'data_test/{step:02d}.jpg', out)

    # if False:
    #     from image_viewer import ImageViewer
    #     def nothing(x):
    #         pass
    
    #     scale_percent = 20 # percent of original size
    #     width = int(img.shape[1] * scale_percent / 100)
    #     height = int(img.shape[0] * scale_percent / 100)
    #     dim = (width, height)
    #     # resize image
    #     img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        
    #     cv2.namedWindow('flood')
    #     cv2.createTrackbar('loDiff', 'flood', 0, 1000, nothing)
    #     cv2.createTrackbar('upDiff', 'flood', 0, 1000, nothing)
    #     cv2.createTrackbar('flags', 'flood', 1, 2, nothing)
    #     cv2.createTrackbar('x', 'flood', 0, img.shape[1]-1, nothing)
    #     cv2.createTrackbar('y', 'flood', 0, img.shape[0]-1, nothing)

    #     mask = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=np.uint8)

    #     while(1):
    #         k = cv2.waitKey(1) & 0xFF
    #         if k == 27:
    #             break
    #         # get current positions of four trackbars
    #         loDiff = cv2.getTrackbarPos('loDiff', 'flood')
    #         upDiff = cv2.getTrackbarPos('upDiff', 'flood')
    #         flags = cv2.getTrackbarPos('flags', 'flood')
    #         x = cv2.getTrackbarPos('x', 'flood')
    #         y = cv2.getTrackbarPos('y', 'flood')
    #         flags *= 4
    #         # loDiff /= 100
    #         # upDiff /= 100

    #         rect = cv2.floodFill(img.copy(), mask, (x, y), (0, 0, 0), loDiff=loDiff, upDiff=upDiff, flags=flags | ( 255 << 8 ))
    #         cv2.imshow('flood', rect[2])
    #     cv2.destroyAllWindows()
    
    # from magicwand import SelectionWindow
    # window = SelectionWindow(img)
    # window.show()