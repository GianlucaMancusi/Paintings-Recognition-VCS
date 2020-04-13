import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def mean_shift_segmentation(img : np.array,spatial_radius=7, color_radius=30, maximum_pyramid_level=1):
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

if __name__ == "__main__":
    rgbImage = cv2.imread('dataset/photos/001/GOPR5832/000030.jpg')
    # rgbImage = cv2.imread('data_test/gallery_0.jpg')
    meanshiftseg = mean_shift_segmentation(rgbImage)


    f, axarr = plt.subplots(1, 2)
    rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)
    meanshiftseg = cv2.cvtColor(meanshiftseg, cv2.COLOR_BGR2RGB)
    axarr[0].imshow(rgbImage)
    axarr[1].imshow(meanshiftseg)
    plt.show()
    pass