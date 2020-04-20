import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def hough(previousOutput):
    """
    Return the lines found in the image
    Parameters
    ----------
    previousOutput : np.array
        image in grayscale or black and white form

    Returns
    -------
    list
        list of all lines found in the image, None if no image is found
    """
    return cv2.HoughLines(previousOutput, 1, np.pi / 180, 35, None, 0, 0)

def main():
    rgbImage = cv2.imread('data_test/08_edges.png')
    grayImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2GRAY)
    _, previousOutput = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    cdst = cv2.cvtColor(previousOutput, cv2.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    lines = hough(previousOutput)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    linesP = cv2.HoughLinesP(previousOutput, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)
    
    f, axarr = plt.subplots(1, 3)
    axarr[0].imshow(previousOutput, cmap='gray')
    axarr[1].imshow(cdst)
    axarr[2].imshow(cdstP)
    plt.show()


if __name__ == '__main__':
    main()