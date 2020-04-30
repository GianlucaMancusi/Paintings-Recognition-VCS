import numpy as np
import cv2
from stopwatch import Stopwatch
from data_test.standard_samples import TEST_PAINTINGS, RANDOM_PAINTING


def filter_matches(matches, threshold: float):
    correct_matches = []
    for i, m in enumerate(matches):
        if i < len(matches) - 1 and m.distance < 0.7 * matches[i+1].distance:
            correct_matches.append(m)
    return correct_matches


def draw_matches(matches, img1, img2, kp1, kp2, threshold=0.7):
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                          sorted_matches[:10], None, flags=2)
    return img3


def get_flann_matcher():
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def resize_image(img, resize_factor: float):
    return cv2.resize(img, (int(
        img.shape[0] * resize_factor), int(img.shape[1] * resize_factor)), interpolation=cv2.INTER_AREA)


def retrieve_painting(painting: str, dataset: list, threshold=0.7, neighbors=2, verbose=False):
    
    # install opencv-contrib python
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img1 = cv2.imread(painting, cv2.IMREAD_GRAYSCALE)

    # since FHD images are too heavy, they're simply brutally resized.
    # The number of matching keypoints  appers to scale according to this factor
    resize_factor = 0.15
    img1 = resize_image(img1, resize_factor)

    kp1, des1 = orb.detectAndCompute(img1, None)
    matches_counts = []
    for p in dataset:
        img2 = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img2 = resize_image(img2, resize_factor)

        kp2, des2 = orb.detectAndCompute(img2, None)
        matches = bf.match(des1, des2,)
        matches = filter_matches(matches, threshold=threshold)
        matches_counts.append(len(matches))
        if verbose:
            print("Score between {} and {}: {}".format(
                painting, p, len(matches)))
            cv2.imshow("Comparison between {} and {}".format(painting, p),
                      draw_matches(matches, img1, img2, kp1, kp2))

    return dataset[np.argmax(matches_counts)]


if __name__ == "__main__":
    dataset = TEST_PAINTINGS
    painting = TEST_PAINTINGS[4]
    dataset.remove(painting)

    watch = Stopwatch()
    verbose = True
    watch.start()
    res = retrieve_painting(painting, dataset, verbose=verbose)
    print(watch.stop())
    if verbose:
        cv2.waitKey(0)
