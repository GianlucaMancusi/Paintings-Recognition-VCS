import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from step_06_corners_detection import hough
from step_06_corners_detection import find_corners


def mask(img, corners):
    """
    Given an image and the four corners of the picture it returns an image with
    the same shape and with the mask drawn.
    
    Parameters
    ----------
    img : np.array
        image where the polygon will be drawn
    corners : list
        list of corners [x, y]

    Returns
    -------
    image
        image in binary where there is drawn the mask
    """
    polyImg = np.zeros_like(img)

    pts = np.array(corners, np.int32)
    pts = cv2.convexHull(pts)
    pts = pts.reshape((-1,1,2))

    cv2.fillPoly(polyImg, [pts], 255)
    return polyImg


def main():
    rgbImage = cv2.imread('data_test/08_edges.png')
    grayImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
    _, cannyOutput = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    previousOutput = hough(cannyOutput)


    corners = find_corners(previousOutput) 
    cornersImg = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2RGB)  
    for point in corners:
        cv2.circle(cornersImg,(point[0], point[1]), 6, (0,255,0), -1)   

    mask_img = mask(grayImage, corners)

    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(grayImage, cmap='gray')
    axarr[0].set_title('Source')
    axarr[1].imshow(cornersImg)
    axarr[1].set_title('The 4 corners found')
    axarr[2].imshow(mask_img, cmap='gray')
    axarr[2].set_title('Mask created')
    plt.show()

if __name__ == '__main__':
    main()