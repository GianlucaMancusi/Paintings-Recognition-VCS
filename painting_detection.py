import cv2
from step_01_mean_shift_seg import _mean_shift_segmentation
from step_02_mask_largest_segment import _mask_largest_segment
from step_03_dilate_invert import _erode_dilate, _invert, _add_padding

def painting_detection(img):
    out = _mean_shift_segmentation(img)
    out = _mask_largest_segment(out)
    return out

if __name__ == "__main__":
    from data_test.standard_samples import TEST_PAINTINGS, PEOPLE
    from image_viewer import ImageViewer
    from stopwatch import Stopwatch

    iv = ImageViewer(cols=3)
    watch = Stopwatch()
    for filename in (PEOPLE, ):
        img = cv2.imread(filename)
        watch.start()
        res = painting_detection(img)
        iv.add(res, cmap='bgr', title=f'time={watch.stop():.02f}s')
    iv.show()
