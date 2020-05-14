import numpy as np
import cv2
from stopwatch import Stopwatch
from data_test.standard_samples import TEST_PAINTINGS, RANDOM_PAINTING, PAINTINGS_DB


def filter_matches(matches, threshold=35):
    length, correct_matches = len(matches), []
    for i, m in enumerate(matches):
        # if i < length - 1 and m.distance < threshold * matches[i+1].distance:
        if m.distance < threshold:
            correct_matches.append(m)
    # import matplotlib.pyplot as plt
    # dist = [m.distance for m in matches]
    # plt.hist(dist, bins=25)
    # plt.gca().set(title=f'len(correct_matches)={len(correct_matches)}')
    # plt.show()
    return correct_matches


def draw_matches(matches, img1, img2, kp1, kp2,):
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                           sorted_matches[:10], None, flags=2)
    return img3


def resize_image(img, resize_factor: float):
    return cv2.resize(img, (int(
        img.shape[0] * resize_factor), int(img.shape[1] * resize_factor)), interpolation=cv2.INTER_AREA)


def retrieve_painting(painting, dataset, threshold=35, resize_factor=0.10, verbose=False):
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
    list
        returns a normalized histogram containing the confindence of each dataset's painting to contain the target painting 
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

        # al posto di calcolarlo ogni volta si potrebbe salvare des2 da qualche parte
        kp2, des2 = orb.detectAndCompute(img2, None)
        if des2 is None:
            matches_counts.append(0)
        else:
            matches = bf.match(des1, des2)
            matches = filter_matches(matches, threshold=threshold)
            matches_counts.append(len(matches))
        if verbose:
            print(f"The image {i + 1} shares {len(matches)} keypoints")
            cv2.imshow(f"Comparison with image {i + 1}",
                       draw_matches(matches, img1, img2, kp1, kp2))

    summation = np.sum(matches_counts)
    m_np = np.array(matches_counts)
    # return [m / m_np.max() for m in matches_counts]
    return matches_counts

def best_match(scores):
    np_scores = np.array(scores)
    top_2 = np_scores.argsort()[-2:][::-1]
    ratio = 'inf'
    if scores[top_2[1]] != 0:
        ratio = scores[top_2[0]] - scores[top_2[1]]
    return top_2[0], ratio

if __name__ == "__main__":
    # from image_viewer import ImageViewer
    # watch = Stopwatch()

    # test_image_index = 4

    # dataset = PAINTINGS_DB
    # painting = TEST_PAINTINGS[test_image_index]

    # dataset_images = [cv2.imread(url, 0) for url in dataset]
    # painting_image = cv2.imread(painting, 0)

    # top_of_values = 5
    # iv = ImageViewer(cols=3)
    # iv.add(painting_image, cmap='bgr', title='source')

    # verbose = False
    # watch.start()
    # scores = retrieve_painting(painting_image, dataset_images, verbose=verbose, resize_factor=0.8)
    # scores_np = np.array(scores)
    # top_args = scores_np.argsort()[-top_of_values:][::-1]
    # print(f'top-{top_of_values} options:')
    # for i, idx in enumerate(top_args):
    #     print(f'\t[{idx}]\t --> "{dataset[idx]}"\t score={scores_np[idx]}')
    #     iv.add(dataset_images[idx], cmap='bgr', title=f'[{i + 1}] {scores_np[idx]}')
    # res = top_args[0]
    # print(watch.stop())
    # print(f"The painting in the image {painting} is also contained in image {dataset[res]}")
    # iv.show()
    # if verbose:
    #     cv2.waitKey(0)
    

    from pipeline import Pipeline, Function
    from image_viewer import ImageViewer
    from data_test.standard_samples import RANDOM_PAINTING, PAINTINGS_DB, TEST_RETRIEVAL
    from painting_rectification import four_point_transform, remove_pad
    from painting_retrieval import retrieve_painting, best_match
    dataset_images = [cv2.imread(url) for url in PAINTINGS_DB]
    for painting_path in TEST_RETRIEVAL:
        img = cv2.imread(painting_path)
        pipeline = Pipeline()
        pipeline.set_default(10)
        list_corners = pipeline.run(img, filename=painting_path)
        list_corners = [ remove_pad(corners, 100) for corners in list_corners ]
        iv = ImageViewer(cols=4)
        iv.remove_axis_values()
        for i, corners in enumerate(list_corners):
            img_sec = four_point_transform(img, np.array(corners))
            if not img_sec is None:
                # img_gray = cv2.cvtColor(img_sec, cv2.COLOR_BGR2GRAY)
                scores = retrieve_painting(img_sec, dataset_images, verbose=False, resize_factor=0.8)
                res, ratio = best_match(scores)
                target = dataset_images[res]
                iv.add(img_sec, cmap='bgr')
                iv.add(target, cmap='bgr', title=f'[{res}] {scores[res]} ratio={ratio}')
        iv.add(img, title='Source', cmap='bgr')
        iv.show()
