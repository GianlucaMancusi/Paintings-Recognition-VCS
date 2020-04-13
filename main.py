import numpy as np
import cv2
import matplotlib.pyplot as plt
from step_01_mean_shift_seg             import *
from step_02_mask_largest_segment       import *
from step_03_dilate_invert              import *
from step_04_connected_components       import *
from step_05_find_paintings             import *
from step_06_erosion                    import *
from step_07_median_filter              import *
from step_08_canny_edge_detection       import *
from step_09_hough                      import *
from step_10_find_corners               import *
from step_11_highlight_painting         import *
from step_12_create_mask                import *

if __name__ == "__main__":
    # rgbImage = cv2.imread('dataset/photos/003/GOPR1935/000015.jpg') # non funziona niente N < K
    rgbImage = cv2.imread('dataset/photos/001/GOPR5832/000030.jpg') # non segmenta bene lo sfondo ma funziona
    # rgbImage = cv2.imread('dataset/photos/014/VID_20180529_113230/000330.jpg') # non funziona niente N < K
    # rgbImage = cv2.imread('dataset/photos/014/VID_20180529_112739/000060.jpg') # da aggiungere il padding nero
    # rgbImage = cv2.imread('dataset/photos/012/IMG_4085/000060.jpg') # errore non trova lo sfondo
    out_rgb = rgbImage.copy()
    s01 = mean_shift_segmentation(rgbImage.copy(), spatial_radius=7, color_radius=150, maximum_pyramid_level=1)
    s02 = mask_largest_segment(s01.copy(), delta=48)
    s03 = erode_dilate(s02.copy(), size=5, erode=False)
    s04 = invert(s03.copy())

    s05 = findContours(s04.copy())
    s05_2 = findPossibleContours(s04, s05)
    s05_debug = s04.copy()
    s05_debug = cv2.cvtColor(s05_debug, cv2.COLOR_GRAY2BGR)
    s06_total = np.zeros_like(s05_debug)
    s06_total_2 = np.zeros_like(s05_debug)
    s07_total = np.zeros_like(s05_debug)
    s08_total = np.zeros_like(s05_debug)[:, :, 1]
    for painting_contour in s05_2:
        color1 = (list(np.random.choice(range(256), size=3)))
        color = [int(color1[0]), int(color1[1]), int(color1[2])]
        cv2.fillPoly(s05_debug, pts=[painting_contour], color=color)

        s06_pre_bw = np.zeros_like(s05_debug)
        cv2.fillPoly(s06_pre_bw, pts=[painting_contour], color=(255,255,255))
        cv2.fillPoly(s06_total, pts=[painting_contour], color=(255,255,255))

        s06_cleaned = clean_frames_noise(s06_pre_bw)
        s06_total_2 += s06_cleaned
        s07_median = apply_median_filter(s06_cleaned)
        s07_total += s07_median
        s08_canny = apply_edge_detection(s07_median)
        s08_total += s08_canny
        _, s08_canny = cv2.threshold(s08_canny, 127, 255, cv2.THRESH_BINARY)

        s09_lines = hough(s08_canny)
        s09_debug = rgbImage.copy()
        if s09_lines is not None:
            for i in range(0, len(s09_lines)):
                rho = s09_lines[i][0][0]
                theta = s09_lines[i][0][1]
                a = math.cos(theta)
                b = math.sin(theta)
                x0 = a * rho
                y0 = b * rho
                pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
                pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
                cv2.line(s09_debug, pt1, pt2, (255,255,255), 3, cv2.LINE_AA)
        s10_corners = find_corners(s09_lines)
        out_rgb = highlight_painting(out_rgb, s10_corners)

    rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)
    s01 = cv2.cvtColor(s01, cv2.COLOR_BGR2RGB)
    f, axarr = plt.subplots(3, 4)
    axarr[0, 0].imshow(rgbImage)
    axarr[0, 0].set_title('Source')
    axarr[0, 1].imshow(s01)
    axarr[0, 1].set_title('mean_shift')
    axarr[0, 2].imshow(s02)
    axarr[0, 2].set_title('mask_largest_segment')
    axarr[0, 3].imshow(s03)
    axarr[0, 3].set_title('erode_dilate')
    axarr[1, 0].imshow(s04)
    axarr[1, 0].set_title('invert')
    axarr[1, 1].imshow(s05_debug)
    axarr[1, 1].set_title('find_paintings {}'.format(len(s05_2)))
    if len(s05_2) != 0:
       axarr[1, 2].imshow(s06_total)
       axarr[1, 2].set_title('last_contuour')
       axarr[1, 3].imshow(s06_total_2)
       axarr[1, 3].set_title('last_cleaned')
       axarr[2, 0].imshow(s07_total)
       axarr[2, 0].set_title('last_median')
       axarr[2, 1].imshow(s08_total)
       axarr[2, 1].set_title('last_canny')
       axarr[2, 2].imshow(s09_debug)
       axarr[2, 2].set_title('last_lines')
       axarr[2, 3].imshow(out_rgb)
       axarr[2, 3].set_title('result')
    # for ax in axarr:
    #     ax.xticks([])
    #     ax.yticks([])
    plt.show()