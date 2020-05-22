import cv2
import numpy as np
import math

def draw_lines(img, lines, pad):
    canvas = np.stack((img,)*3, axis=-1)
    color = np.random.randint(256, size=3).tolist()
    if not lines is None: 
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)) - pad, int(y0 + 1000*(a)) - pad)
            pt2 = (int(x0 - 1000*(-b)) - pad, int(y0 - 1000*(a)) - pad)
            cv2.line(canvas, pt1, pt2, color, 3, cv2.LINE_AA)
    return canvas

def _hough(img):
    """
    Return the lines found in the image
    Parameters
    ----------
    img : np.array
        image in grayscale or black and white form

    Returns
    -------
    list
        list of all lines found in the image, None if no image is found
    """
    lines = cv2.HoughLines(img, 1, np.pi / 180, 40, None, 0, 0)
    return lines

def hough(input, pad=0, debug=False):
    lines = _hough(input)
    if debug:
        canvas = draw_lines(input, lines, pad)
        return (input, lines), canvas
    else:
        return (input, lines)

if __name__ == '__main__':
    from pipeline import Pipeline
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread(RANDOM_PAINTING)
    pipeline = Pipeline()
    pipeline.set_default(9)
    pipeline.run(img, debug=True, print_time=True, filename=RANDOM_PAINTING)
    pipeline.debug_history().show()