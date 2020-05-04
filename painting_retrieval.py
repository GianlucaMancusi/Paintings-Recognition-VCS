import numpy as np
import cv2
from stopwatch import Stopwatch
from data_test.standard_samples import TEST_PAINTINGS, RANDOM_PAINTING


def filter_matches(matches, threshold: float):
    length, correct_matches = len(matches), []
    for i, m in enumerate(matches):
        if i < length - 1 and m.distance < threshold * matches[i+1].distance:
            correct_matches.append(m)
    return correct_matches


def draw_matches(matches, img1, img2, kp1, kp2,):
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


def retrieve_painting(painting, dataset, threshold=0.7, resize_factor=0.10, verbose=False):
    """
    Finds a given painting inside a dataset.

    Parameters
    ----------
    painting : np.array
        image to find inside the dataset
    dataset : list of np.array
        list of the images to analyze
    threshold: float
        the minimum ratio that a pair of keypoints must have to be considered as a pair of matching points  
    resize_factor: float
        images' sizes is resized by this factor to speed up computation 
    verbose: bool
        if True, the results are shown on the console and on the images

    Returns
    -------
    int
        returns the index of the image of the dataset having the highest number of matching points
    """

    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img1 = painting
    img1 = resize_image(img1, resize_factor)

    kp1, des1 = orb.detectAndCompute(img1, None)
    matches_counts = []
    for i, p in enumerate(dataset):
        img2 = p
        img2 = resize_image(img2, resize_factor)

        kp2, des2 = orb.detectAndCompute(img2, None)
        matches = bf.match(des1, des2,)
        matches = filter_matches(matches, threshold=threshold)
        matches_counts.append(len(matches))
        if verbose:
            print(f"The image {i + 1} shares {len(matches)} keypoints")
            cv2.imshow(f"Comparison with image {i + 1}",
                       draw_matches(matches, img1, img2, kp1, kp2))

    return np.argmax(matches_counts)


if __name__ == "__main__":
    watch = Stopwatch()

    test_image_index = 4

    dataset = TEST_PAINTINGS
    painting = TEST_PAINTINGS[test_image_index]
    dataset.remove(painting)

    dataset_images = [cv2.imread(url, 0) for url in TEST_PAINTINGS]
    painting_image = cv2.imread(painting, 0)

    verbose = False
    watch.start()
    res = retrieve_painting(painting_image, dataset_images, verbose=verbose)
    print(watch.stop())
    print(
        f"The painting in the image {painting} is also contained in image {dataset[res]}")
    if verbose:
        cv2.waitKey(0)
