import cv2
import matplotlib.pyplot as plt
from step_06_corners_detection import hough
from step_05_contour_pre_processing import _apply_edge_detection
from skimage.transform import hough_line
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from generate_painting_masks import generate_mask
from scipy.ndimage import gaussian_filter1d
from step_03_cleaning import _add_padding


def nothing(x):
    pass


def _undistort(img, k1=0, k2=0, fx=857, fy=876):
    img = img.copy()
    # K = np.array([[5.00*img.shape[1],     0.,  img.shape[1]//2],
    #              [0.,   5.00*img.shape[0],   img.shape[0]//2],
    #              [0.,     0.,     1.]])
    K = np.array([[fx,     0.,     img.shape[1]//2],
                  [0.,     fy,     img.shape[0]//2],
                  [0.,     0.,     1.]])

    # zero distortion coefficients work well for this image
    D = np.array([k1, k2, 0., 0.])

    # use Knew to scale the output
    Knew = K.copy()
    Knew[(0, 1), (0, 1)] = .6 * Knew[(0, 1), (0, 1)]
    img_undistorted = cv2.undistort(img, K, D, newCameraMatrix=Knew)
    return img_undistorted


def undistort(img, **kwargs):
    out = _undistort(img, **kwargs)
    # kwargs['k1'] = 0
    # kwargs['k2'] = 0
    # s1, e1, s2, e2 = calculate_crop_spot(_undistort(img, **kwargs))
    # out = out[s2:e2, s1:e1]
    out = without_black_pad(out)
    out = cv2.resize(out, (img.shape[1], img.shape[0]))
    return out


def threshold_undistort(img, crop_spot=None, **kwargs):
    out = _undistort(img, **kwargs)
    if crop_spot:
        s1, e1, s2, e2 = crop_spot
        out = out[s2:e2, s1:e1]
    return out


def apply_fisheye(img, debug=False):
    base_img = img.copy()

    if not debug:
        img_undistorted = undistort(base_img, k1=-0.38, k2=0.106, fx=872, fy=872)
    else:
        cv2.namedWindow('undistorted')
        cv2.createTrackbar('k1', 'undistorted', 0, 10000, nothing)
        # cv2.createTrackbar('k2', 'undistorted', 0, 1000, nothing)
        # cv2.createTrackbar('fx', 'undistorted', 0, 5500, nothing)
        cv2.createTrackbar('fy', 'undistorted', 0, 5500, nothing)
        while(1):
            # get current positions of four trackbars
            k1 = cv2.getTrackbarPos('k1', 'undistorted')
            # k2 = cv2.getTrackbarPos('k2', 'undistorted')
            # fx = cv2.getTrackbarPos('fx', 'undistorted')
            fy = cv2.getTrackbarPos('fy', 'undistorted')
            k1 = k1/10000*20 - 10
            k2 = k1
            fx = fy

            if debug:
                print(k1, fx)

            img_undistorted = undistort(base_img, k1=k1, k2=k2, fx=fx, fy=fy)
            if debug:
                cv2.imshow('undistorted', img_undistorted)
                
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
    return img_undistorted

def vertical_lines(half_freedom=1):
    angles = np.arange(0, np.pi, np.pi/180)
    a = angles[:half_freedom]
    b = angles[-half_freedom:]
    return np.concatenate((a, b))

def horizontal_lines(half_freedom=1):
    angles = np.arange(0, np.pi, np.pi/180)
    return angles[90 - half_freedom: 90 + half_freedom]

def hv_lines(half_freedom=1):
    return np.concatenate((vertical_lines(half_freedom), horizontal_lines(half_freedom)))

def calculate_HT(img, sigma=3):
    angles = hv_lines(10)
    H, _, _ = hough_line(img, angles)
    H = gaussian_filter1d(H, sigma)
    return H.max()

def HTRDC(img, steps=50, range_min=-0.25, range_max=0, epsilon=0.0001, iteration=0, prev_k=None, debug=False):
    step = (range_max - range_min)/steps
    K_range = np.arange(range_min, range_max, step)
    HT_vals = np.zeros_like(K_range)
    HT_vals_focus = np.zeros_like(K_range)
    crop_spot = calculate_crop_spot(threshold_undistort(img))
    for i, k in enumerate(K_range):
        img_undistorted = threshold_undistort(img, crop_spot=crop_spot, k1=k)
        HT = calculate_HT(img_undistorted)
        HT_vals_focus[i] = HT
        img_undistorted = threshold_undistort(img, k1=k)
        HT = calculate_HT(img_undistorted)
        HT_vals[i] = HT
        
        if debug:
            cv2.imshow('undistorted', img_undistorted)
            cv2.waitKey(1)
            print(f'k={k:.08f} HT={HT}')

    smoothed_HT_vals = gaussian_filter1d(HT_vals, 1.5)
    # plt.plot(K_range, HT_vals, label='HT')
    # plt.plot(K_range, HT_vals_focus, label='HT_focused')
    # plt.legend()
    # # plt.plot(K_range, smoothed_HT_vals)
    # plt.show()
    if debug:
        print(f'k={K_range[HT_vals.argmax()]} k_smooth={K_range[smoothed_HT_vals.argmax()]}')
    max_k = K_range[HT_vals_focus.argmax()]
    if iteration != 0 and (prev_k - max_k) <= epsilon:
        return max_k
    else:
        range_min = max(range_min, max_k - step)
        range_max = min(max_k + step, range_max)
        return HTRDC(img, steps, range_min, range_max, epsilon, iteration + 1, max_k)


def distort_random(img):
    return undistort(img, k1=-0.5, k2=-0.5)

def without_black_pad(img):
    s1, e1, s2, e2 = calculate_crop_spot(img)
    return img[s2:e2, s1:e1]

def calculate_crop_spot(img):
    img = img.copy()
    if img.ndim == 3:
        img = img.sum(axis=2)
    assert img.ndim == 2
    width_max = img.max(axis=0)
    height_max = img.max(axis=1)
    s1, e1 = args_start_end_sim(width_max)
    s2, e2 = args_start_end_sim(height_max)
    return (s1, e1, s2, e2)

def args_start_end(arr):
    assert arr.ndim == 1
    start = start_at(arr)
    end = arr.shape[0] - start_at(arr[::-1])
    return start, end

def args_start_end_sim(arr):
    assert arr.ndim == 1
    start = start_at(arr)
    end = start_at(arr[::-1])
    res = min(start, end)
    return res, arr.shape[0] - res

def start_at(arr):
    assert arr.ndim == 1
    for i, val in enumerate(arr):
        if val != 0:
            break
    return i


if __name__ == "__main__":
    from image_viewer import show_me, ImageViewer
    from data_test.standard_samples import RANDOM_PAINTING, FISH_EYE, CHESSBOARD, TEST_DISTORTION, get_random_paintings
    for paint in [FISH_EYE, ]:
        img = cv2.imread(paint)
        img = cv2.resize(img, (1280, 720))
        # img = distort_random(img)
        iv = ImageViewer()
        # img = cv2.imread("data_test/000090.jpg", cv2.IMREAD_GRAYSCALE)
        # for e in range(0,100,5):
        mask = generate_mask(img)
        iv.add(mask, cmap='bgr', title='mask')
        
        canvas = np.empty_like(mask, dtype=np.float32)
        # canvas += _apply_edge_detection(mask, 150, 250)
        canvas += _apply_edge_detection(img, 150, 250)
        canvas = np.clip(canvas, 0, 255)
        iv.add(img, cmap='bgr', title='original')
        iv.add(canvas, cmap='bgr', title='canvas')

        # apply_fisheye(img, True)

        # vals_1 = []
        # vals_2 = []
        # pads = range(0, 500, 5)
        # for pad in pads:
        #     tmp = canny.copy()
        #     tmp = _add_padding(tmp, pad)
        #     # tmp = cv2.resize(tmp, (1280, 720))
        #     HT = calculate_HT(tmp)
        #     vals_1.append(HT)
        #     tmp = without_black_pad(tmp)
        #     HT = calculate_HT(tmp)
        #     vals_2.append(HT)
        #     cv2.imshow('tmp', tmp)
        #     cv2.waitKey(1)
        #     print(f'pad={pad} HT={vals_1[-1]} R_HT={vals_2[-1]}')
        # plt.plot(pads, vals_1)
        # plt.plot(pads, vals_2)
        # plt.show()

        k = HTRDC(canvas, steps=50, range_min=-0.25, range_max=0)
        canvas = undistort(img, k1=k)
        # canvas = undistort(img, k1=-0.38, k2=0.106, fx=872, fy=872)
        cv2.imwrite('data_test/00_calibration_desired.jpg', canvas)
        # canvas = undistort(img, k1=-0.257, k2=0.008)
        iv.add(canvas, cmap='bgr', title=f'{paint} k={k:.06f}')
        iv.show()
        # show_me(canvas)
