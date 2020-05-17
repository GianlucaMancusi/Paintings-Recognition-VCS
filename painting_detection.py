import cv2
import numpy as np
from step_01_mean_shift_seg import _mean_shift_segmentation
from step_02_mask_largest_segment import _mask_largest_segment
from step_03_dilate_invert import _erode_dilate, _invert, _add_padding
from step_04_connected_components import _find_contours
from step_05_find_paintings import _find_possible_contours
from step_06_erosion import  _clean_frames_noise, _mask_from_contour
from step_07_median_filter import _apply_median_filter
from step_08_canny_edge_detection import _apply_edge_detection
from step_09_hough import _hough
from step_10_find_corners import _find_corners
from step_11_highlight_painting import _highlight_paintings, _draw_all_contours
from step_12_b_create_outer_rect import _mask, draw_rect, rect_contour

def painting_detection(img, pad=1, area_perc=0.93):
    out = _mean_shift_segmentation(img)
    out = _mask_largest_segment(out)
    out = _erode_dilate(out)
    out = _invert(out)
    out = _add_padding(out, pad)
    contours = _find_contours(out)
    contours = _find_possible_contours(out, contours, min_area_percentage=0.7, min_width=150, min_height=100)
    paintings_contours = []
    for contour in contours:
        _, _, w, h = cv2.boundingRect(contour)
        if cv2.contourArea(contour) / (w * h) > area_perc:
            paintings_contours.append(rect_contour(contour, pad))
        else:
            for_out = _mask_from_contour(out, contour)
            for_out = _clean_frames_noise(for_out)
            for_out = _apply_median_filter(for_out)
            for_out = _apply_edge_detection(for_out)
            # canvas += for_out
            lines = _hough(for_out)
            if not lines is None:
                corners = _find_corners(lines)
                pts = np.array(corners, np.int32)
                pts = cv2.convexHull(pts)
                pts = pts.reshape((-1,1,2))
                pts_ratio = cv2.contourArea(contour) / (cv2.contourArea(pts) + 1)
                if pts_ratio < 1.2 and pts_ratio > cv2.contourArea(contour) / (w * h):
                    paintings_contours.append(pts)
                else:
                    paintings_contours.append(rect_contour(contour, pad))
            else:
                paintings_contours.append(rect_contour(contour, pad))
    return paintings_contours

if __name__ == "__main__":
    from data_test.standard_samples import TEST_PAINTINGS, PEOPLE, TEST_RETRIEVAL
    from image_viewer import ImageViewer, show_me
    from stopwatch import Stopwatch

    iv = ImageViewer(cols=3)
    watch = Stopwatch()
    for filename in TEST_RETRIEVAL:
        img = cv2.imread(filename)
        watch.start()
        paintigs_contours = painting_detection(img)
        res = _draw_all_contours(paintigs_contours, img)
        time = watch.stop()
        iv.add(res, cmap='bgr', title=f'time={time:.02f}s')
        print(f'filename={filename} time={time:.02f}s')
        show_me(res, title=f'filename={filename} time={time:.02f}s n={len(paintigs_contours)}')
    iv.show()
