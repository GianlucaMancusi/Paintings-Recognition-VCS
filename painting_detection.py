import cv2
import numpy as np
from step_01_pre_processing import _mean_shift_segmentation
from step_02_background_detection import _mask_largest_segment
from step_03_cleaning import _opening, _invert, _add_padding
from step_04_components_selection import _find_contours
from step_04_components_selection import _find_possible_contours
from step_05_contour_pre_processing import  _clean_frames_noise, _mask_from_contour
from step_05_contour_pre_processing import _apply_median_filter
from step_05_contour_pre_processing import _apply_edge_detection
from step_06_corners_detection import _hough
from step_06_corners_detection import _find_corners
from step_07_highlight_painting import _highlight_paintings, _draw_all_contours
from step_08_b_create_outer_rect import _mask, draw_rect, rect_contour

def painting_detection(img, pad=1, area_perc=0.93):
    out = _mean_shift_segmentation(img)
    out = _mask_largest_segment(out)
    out = _opening(out)
    out = _invert(out)
    out = _add_padding(out, pad)
    contours = _find_contours(out)
    contours = _find_possible_contours(out, contours)
    paintings_contours = []
    # canvas = np.zeros_like(out)
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        found_correct_shape = False
        for_out = _mask_from_contour(out, contour)
        # show_me(for_out)
        for_out = _clean_frames_noise(for_out)
        for_out = _apply_median_filter(for_out)
        for_out = _apply_edge_detection(for_out)
        # canvas += for_out
        # show_me(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) * _mask_from_contour(out, approx)[pad:-pad, pad:-pad])
        lines = _hough(for_out)
        if lines is not None:
            corners = _find_corners(lines)
            if corners is not None:
                pts = np.array(corners, np.int32)
                pts = cv2.convexHull(pts)
                pts = pts.reshape((-1,1,2))
                pts_ratio = cv2.contourArea(contour) / (cv2.contourArea(pts) + 1)
                if pts_ratio < 1.2 and pts_ratio > cv2.contourArea(contour) / (w * h):
                    paintings_contours.append(pts)
                    found_correct_shape = True
            if not found_correct_shape:
                epsilon = 0.1 * cv2.arcLength(contour,True)
                approx = cv2.approxPolyDP(contour, epsilon, closed=True)
                if len(approx) == 4:
                    paintings_contours.append(approx)  #UnboundLocalError: local variable 'pts' referenced before assignment
                    found_correct_shape = True

        if not found_correct_shape:
            paintings_contours.append(rect_contour(contour, pad))
    return paintings_contours

if __name__ == "__main__":
    from data_test.standard_samples import TEST_PAINTINGS, PEOPLE, TEST_RETRIEVAL, PERSPECTIVE, get_random_paintings
    from image_viewer import ImageViewer, show_me
    from stopwatch import Stopwatch
    from step_06_corners_detection import draw_lines

    iv = ImageViewer(cols=3)
    watch = Stopwatch()
    for filename in ['data_test/painting_09/00_calibration.jpg',]:
        img = cv2.imread(filename)
        # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        watch.start()
        paintigs_contours = painting_detection(img)
        res = _draw_all_contours(paintigs_contours, img)
        time = watch.stop()
        iv.add(res, cmap='bgr', title=f'time={time:.02f}s')
        print(f'filename={filename} time={time:.02f}s')
        show_me(res, title=f'filename={filename} time={time:.02f}s n={len(paintigs_contours)}')
    iv.show()
