import numpy as np
import cv2
import time
from tools.ImageViewer import ImageViewer

from step_01_mean_shift_seg import *
from step_02_mask_largest_segment import *
from step_03_dilate_invert import *
from step_04_connected_components import *
from step_05_find_paintings import *
from step_06_erosion import *
from step_07_median_filter import *
from step_08_canny_edge_detection import *
from step_09_hough import *
from step_10_find_corners import *
from step_11_highlight_painting import *
from step_12_create_mask import *
from tools.Stopwatch import Stopwatch

class Pipeline:

    def run(self, rgbImage: np.ndarray, last_step: int):
        out_rgb = rgbImage.copy()

        total = Stopwatch()
        stopwatch = Stopwatch()
        
        s01 = mean_shift_segmentation(rgbImage.copy(), spatial_radius=7, color_radius=150, maximum_pyramid_level=1)
        stopwatch.round('step 01')
        if last_step == 1:
            return s01

        s02 = mask_largest_segment(s01.copy(), delta=48, x_samples=30)
        stopwatch.round('step 02')
        if last_step == 2:
            return s02

        s03 = erode_dilate(s02.copy(), size=5, erode=False)
        stopwatch.round('step 03')
        if last_step == 3:
            return s03

        s04 = invert(s03.copy())
        stopwatch.round('step 04')
        if last_step == 4:
            return s04

        bordersize = 100
        s04 = cv2.copyMakeBorder(
                    s04,
                    top=bordersize,
                    bottom=bordersize,
                    left=bordersize,
                    right=bordersize,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[0, 0, 0]
        )

        s05 = findContours(s04.copy())
        s05_2 = findPossibleContours(s04, s05)
        s05_debug = s04.copy()
        s05_debug = cv2.cvtColor(s05_debug, cv2.COLOR_GRAY2BGR)
        stopwatch.round('step 05')
        if last_step == 5:
            return s05_debug

        # initilizing structures
        s06_total = np.zeros_like(s05_debug)
        s06_total_2 = np.zeros_like(s05_debug)
        s07_total = np.zeros_like(s05_debug)
        s08_total = np.zeros_like(s05_debug)[:, :, 1]
        
        for painting_contour in s05_2:
            color1 = (list(np.random.choice(range(256), size=3)))
            color = [int(color1[0]), int(color1[1]), int(color1[2])]
            cv2.fillPoly(s05_debug, pts=[painting_contour], color=color)

            s06_pre_bw = np.zeros_like(s05_debug)
            cv2.fillPoly(s06_pre_bw, pts=[
                         painting_contour], color=(255, 255, 255))
            cv2.fillPoly(s06_total, pts=[
                         painting_contour], color=(255, 255, 255))

            s06_cleaned = clean_frames_noise(s06_pre_bw)
            s06_total_2 += s06_cleaned
            s07_median = apply_median_filter(s06_cleaned)
            s07_total += s07_median
            
            s08_canny = apply_edge_detection(s07_median)
            s08_total += s08_canny
            _, s08_canny = cv2.threshold(
                s08_canny, 127, 255, cv2.THRESH_BINARY)

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
                    pt1 = (int(x0 + 1000*(-b)) - bordersize, int(y0 + 1000*(a)) - bordersize)
                    pt2 = (int(x0 - 1000*(-b)) - bordersize, int(y0 - 1000*(a)) - bordersize)
                    cv2.line(s09_debug, pt1, pt2,
                             (255, 255, 255), 3, cv2.LINE_AA)
            
            s10_corners = find_corners(s09_lines)
            if s10_corners is None:
                continue
            out_rgb = highlight_painting(out_rgb, s10_corners, pad=bordersize)

        total.stop('Total\n')
        return out_rgb

if __name__ == "__main__":
    pipeline = Pipeline()
    smadonne = 9
    # outputs = [cv2.cvtColor(pipeline.run(np.array(cv2.imread("data_test/paintings/"+str(i)+".jpg")), last_step=10), cv2.COLOR_BGR2RGB) for i in range(1, smadonne)]

    iv = ImageViewer(smadonne, cols=3)
    iv.remove_axis_values()
    for i in range(1, smadonne + 1):
        filename = "data_test/paintings/"+str(i)+".jpg"
        print(filename)
        img = cv2.imread(filename)
        out = pipeline.run(np.array(img), last_step=10)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        iv.add(out)
    iv.show()