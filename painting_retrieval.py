import numpy as np
import cv2 as cv
from tools.Stopwatch import Stopwatch


def filter_matches(matches, threshold: float):
    correct_matches = []
    for i, m in enumerate(matches):
        if i < len(matches) - 1 and m.distance < 0.7 * matches[i+1].distance:
            correct_matches.append(m)
    return correct_matches


def draw_matches(matches, img1, img2, kp1, kp2, threshold=0.7):
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(img1, kp1, img2, kp2,
                          sorted_matches[:10], None, flags=2)
    return img3


def get_flann_matcher():
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv.FlannBasedMatcher(index_params, search_params)


def resize_image(img, resize_factor: float):
    return cv.resize(img, (int(
        img.shape[0] * resize_factor), int(img.shape[1] * resize_factor)), interpolation=cv.INTER_AREA)


def retrieve_painting(painting: str, dataset: list, threshold=0.7, neighbors=2, verbose=False):
    
    # install opencv-contrib python
    orb = cv.ORB_create()
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    img1 = cv.imread(painting, cv.IMREAD_GRAYSCALE)

    # since FHD images are too heavy, they're simply brutally resized.
    # The number of matching keypoints  appers to scale according to this factor
    resize_factor = 0.15
    img1 = resize_image(img1, resize_factor)

    kp1, des1 = orb.detectAndCompute(img1, None)
    matches_counts = []
    for p in dataset:
        img2 = cv.imread(p, cv.IMREAD_GRAYSCALE)
        img2 = resize_image(img2, resize_factor)

        kp2, des2 = orb.detectAndCompute(img2, None)
        matches = bf.match(des1, des2,)
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
    verbose = False
    watch.start()
    res = retrieve_painting(painting, dataset, verbose=verbose)
    watch.stop()
    if verbose:
        cv.waitKey(0)
