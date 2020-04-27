import numpy as np
import cv2 as cv
from tools.Stopwatch import Stopwatch


def filter_matches(matches, threshold: float):
    correct_matches = []
    for m in matches:
        if m[0].distance < threshold * m[1].distance:
            correct_matches.append(m)
    return correct_matches


def draw_matches(matches, img1, img2, kp1, kp2, threshold=0.7):
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < threshold * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    return cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)


def get_flann_matcher():
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv.FlannBasedMatcher(index_params, search_params)


def resize_image(img, resize_factor: float):
    return cv.resize(img, (int(
        img.shape[0] * resize_factor), int(img.shape[1] * resize_factor)), interpolation=cv.INTER_AREA)


def find_painting(painting: str, dataset: list, threshold=0.7, neighbors=2, verbose=False):
    # creates SIFT instance: some versions of openCV do not contain that
    # if you have some troubles with it, run the following commands:
    # pip uninstall opencv-python
    # pip uninstall opencv-contrib-python
    # pip install opencv-contrib-python==3.4.2.17

    sift = cv.xfeatures2d.SIFT_create()
    flann = get_flann_matcher()

    img1 = cv.imread(painting, cv.IMREAD_GRAYSCALE)

    # since FHD images are too heavy, they're simply brutally resized.
    # The number of matching keypoints  appers to scale according to this factor
    resize_factor = 0.1
    img1 = resize_image(img1, resize_factor)

    kp1, des1 = sift.detectAndCompute(img1, None)
    matches_counts = []
    for p in dataset:
        img2 = cv.imread(p, cv.IMREAD_GRAYSCALE)
        img2 = resize_image(img2, resize_factor)

        kp2, des2 = sift.detectAndCompute(img2, None)
        matches = flann.knnMatch(des1, des2, k=neighbors)
        matches = filter_matches(matches, threshold=threshold)
        matches_counts.append(len(matches))
        if verbose:
            print("Score between {} and {}: {}".format(
                painting, p, len(matches)))
            cv.imshow("Comparison between {} and {}".format(painting, p),
                      draw_matches(matches, img1, img2, kp1, kp2))

    return dataset[np.argmax(matches_counts)]


if __name__ == "__main__":
    test_image_id = 5
    dataset = ["data_test/paintings/{}.jpg".format(i)
               for i in range(1, 9 + 1) if i != test_image_id]

    painting = "data_test/paintings/{}.jpg".format(test_image_id)

    watch = Stopwatch()
    verbose = True
    watch.start()
    res = find_painting(painting, dataset, verbose=verbose)
    watch.stop()
    if verbose:
        cv.waitKey(0)
