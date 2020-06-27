import cv2
import numpy as np

"""
    Applies Canny's edge detector
    
    Parameters
    ----------
    img
        the image

    Returns
    -------
    img
        the image's edges
"""
def _apply_edge_detection(img, t1=50, t2=100):
    result = cv2.Canny(img, t1, t2)
    return result

def apply_edge_detection(input, debug=False):
    result = _apply_edge_detection(input)
    if debug:
        kernel = np.ones((5, 5), np.uint8)
        debug_img = cv2.dilate(result, kernel)
        return result, debug_img
    else:
        return result


if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING, TEST_PAINTINGS
    from pipeline import Pipeline

    filename = TEST_PAINTINGS[2]
    step = 8

    img = cv2.imread(filename)
    pipeline = Pipeline()
    pipeline.set_default(step)
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)
    pipeline.debug_history().show()
    cv2.imwrite(f'data_test/{step:02d}.jpg', out)
