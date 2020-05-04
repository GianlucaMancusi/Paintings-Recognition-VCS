import cv2
import matplotlib.pyplot as plt
from step_09_hough import hough
from step_08_canny_edge_detection import apply_edge_detection

import numpy as np

k_min = 0
k_max = 1
n = 20


def undistort(img):
    pass


# QUI STAVO IMPLEMENTANDO IL PAPER DELLA CUCCHIARA...
# def findKValue(img):
#     original_ROI = img[:365, :100]
#     c_x, c_y = img.shape[0]//2, img.shape[1]//2

#     new_ROI = np.zeros_like(original_ROI)
#     index_matrix = np.indices(new_ROI.shape)
#     index_matrix[0] = index_matrix[0]-c_x
#     index_matrix[1] = index_matrix[1]-c_y

#     for i in range(n):
#         k1 = k_min + i * (k_max - k_min)/n
#         # distorted radius
#         r_d = np.sqrt(index_matrix[0]**2+index_matrix[1]**2)
#         # undistorted coords (new image)
#         x_u = index_matrix[0] + (index_matrix[0] - c_x)*(k1*r_d**2)
#         y_u = index_matrix[1] + (index_matrix[1] - c_y)*(k1*r_d**2)
#         # undistorted radius
#         r_u = np.sqrt(index_matrix[0]**2+index_matrix[1]**2)
#         if k1 > 0:
#             r_d = np.cbrt(r_u/(2*k1)+np.sqrt((1/(3*k1))**3+(r_u/(2*k1))**2)) + \
#                 np.cbrt(r_u/(2*k1)-np.sqrt((1/(3*k1))**3+(r_u/(2*k1))**2))
#         # distorted coords (original image)
#         x_d = c_x + (index_matrix[0]-c_x)*(r_d/r_u)
#         y_d = c_y + (index_matrix[1]-c_y)*(r_d/r_u)

#         new_ROI = original_ROI[y_d, x_d]

#     # LOOP algorithm


def nothing(x):
    pass


def fisheye(img, k1=0, k2=0, fx=1230, fy=1230):

    img = img.copy()
    # K = np.array([[5.00*img.shape[1],     0.,  img.shape[1]//2],
    #              [0.,   5.00*img.shape[0],   img.shape[0]//2],
    #              [0.,     0.,     1.]])
    K = np.array([[fx,     0.,  img.shape[1]//2],
                  [0.,   fy,   img.shape[0]//2],
                  [0.,     0.,     1.]])

    # zero distortion coefficients work well for this image
    D = np.array([k1, k2, 0., 0.])

    # use Knew to scale the output
    Knew = K.copy()
    Knew[(0, 1), (0, 1)] = .6 * Knew[(0, 1), (0, 1)]
    img_undistorted = cv2.undistort(img, K, D, newCameraMatrix=Knew)
    #img_undistorted = cv2.fisheye.undistortImage(img, K, D=D, Knew=Knew)
    return img_undistorted


def apply_fisheye(img, debug=False):
    base_img = img.copy()

    if not debug:
        img_undistorted = fisheye(base_img, k1=-0.38, k2=0.106, fx=872, fy=872)
    else:
        cv2.namedWindow('undistorted')
        cv2.createTrackbar('k1', 'undistorted', 0, 1000, nothing)
        cv2.createTrackbar('k2', 'undistorted', 0, 1000, nothing)
        cv2.createTrackbar('fx', 'undistorted', 0, 5500, nothing)
        cv2.createTrackbar('fy', 'undistorted', 0, 5500, nothing)
        while(1):
            # get current positions of four trackbars
            k1 = cv2.getTrackbarPos('k1', 'undistorted')
            k2 = cv2.getTrackbarPos('k2', 'undistorted')
            fx = cv2.getTrackbarPos('fx', 'undistorted')
            fy = cv2.getTrackbarPos('fy', 'undistorted')
            k1 = k1/1000*2-.5
            k2 = k2/1000*2-.5

            img_undistorted = fisheye(base_img, k1=k1, k2=k2, fx=fx, fy=fy)
            cv2.imshow('undistorted', img_undistorted)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
    return img_undistorted


if __name__ == "__main__":
    img = cv2.imread("data_test\\fisheye.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1280, 720))
    cv2.imshow("normal", img)
    # img = cv2.imread("data_test\\000090.jpg", cv2.IMREAD_GRAYSCALE)
    #canny = apply_edge_detection(img)
    #cv2.imshow("Canny", canny)
    # cv2.waitKey()
    # findKValue(canny)
    #(himg, _), canvas = hough(canny, debug=True)
    canvas = apply_fisheye(img, debug=False)
    cv2.imshow("canvas", canvas)
    cv2.waitKey()
