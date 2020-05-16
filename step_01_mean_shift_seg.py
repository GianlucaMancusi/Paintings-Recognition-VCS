import cv2
import numpy as np
import math
from sklearn.cluster import KMeans

def mean_shift_segmentation(input : np.array, spatial_radius=7, color_radius=30, maximum_pyramid_level=1, debug=False):
    img = _mean_shift_segmentation(input, spatial_radius, color_radius, maximum_pyramid_level)

    if debug:
        return img, img
    else:
        return img

def _mean_shift_segmentation(img : np.array, spatial_radius=7, color_radius=30, maximum_pyramid_level=1):
    """
    This function takes an image and mean-shift parameters and 
    returns a version of the image that has had mean shift 
    segmentation performed on it
    
    See also: https://docs.opencv.org/2.4/modules/imgproc/doc/filtering.html#pyrmeanshiftfiltering
    Parameters
    ----------
    img : np.array
        image where to find the mean-shift segmentation

    spatial_radius : int
        The spatial window radius.

    color_radius : int
        The color window radius.

    maximum_pyramid_level : int
        Maximum level of the pyramid for the segmentation.
    """
    img = cv2.pyrMeanShiftFiltering(img, spatial_radius, color_radius, maximum_pyramid_level)
    return img

def kmeans(img : np.array, n_colors=3):
    arr = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(arr)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    less_colors = centers[labels].reshape(img.shape).astype('uint8')
    return less_colors

if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING
    from pipeline import Pipeline
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(1)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()