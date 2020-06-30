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
    return correct_matches


def draw_matches(matches, img1, img2, kp1, kp2,):
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                           sorted_matches[:10], None, flags=2)
    return img3


def resize_image(img, resize_factor: float):
    return cv2.resize(img, (int(
        img.shape[0] * resize_factor), int(img.shape[1] * resize_factor)), interpolation=cv2.INTER_AREA)

db_descriptors = []
db_descriptors_allocated = False
def retrieve_painting(painting, dataset, threshold=30, resize_factor=1, verbose=False, mse=False):
    """
    Finds a given painting inside a dataset.

    Parameters
    ----------
    painting : np.array
        image to find inside the dataset
    dataset : list of np.array
        list of the images to analyze
    threshold: int
        the number of the best keypoints (ordered by distance ASC) to consider  
    resize_factor: float
        images' sizes is resized by this factor to speed up computation 
    verbose: bool
        if True, the results are shown on the console and on the images

    Returns
    -------
    list
        returns a histogram containing the confindence of each dataset's painting to contain the target painting 
    """

    db_des_filename = "db_descriptors.npy"
    global db_descriptors
    global db_descriptors_allocated

    orb = cv2.ORB_create(nfeatures=250)
    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    img1 = painting
    #img1 = resize_image(img1, resize_factor)
    img1 = cv2.resize(img1.copy(), (400, 400))

    kp1, des1 = orb.detectAndCompute(img1, None)
    matches_counts = []

    for i, p in enumerate(dataset):
        des2 = None
        if not db_descriptors_allocated:
            img2 = p
            #img2 = resize_image(img2, resize_factor)
            img2 = cv2.resize(img2.copy(), (400,400))
            # al posto di calcolarlo ogni volta si potrebbe salvare des2 da qualche parte

            kp2, des2 = orb.detectAndCompute(img2, None)
            size = img2.shape[0], img2.shape[1]
            db_descriptors.append(des2)
        else:
            des2 = db_descriptors[i]
            size = db_descriptors[i]

        if des2 is None:
            matches_counts.append(0)
        else:
            matches = bf.match(des1, des2)
            #matches = filter_matches(matches, threshold=threshold)
            matches = sorted(matches, key=lambda x: x.distance)

            #distances = [(m.distance**2) if mse else m.distance for m in matches[:threshold]]
            distances = [m.distance**2 for m in matches[:threshold]]
            matches_counts.append(np.mean(distances))
        if verbose:
            print(f"The image {i + 1} shares {len(matches)} keypoints")
            cv2.imshow(f"Comparison with image {i + 1}",
                       draw_matches(matches, img1, img2, kp1, kp2))

    if not db_descriptors_allocated:
        db_descriptors_allocated = True

    return matches_counts


def best_match(scores):
    np_scores = np.array(scores)
    top_2 = np_scores.argsort()[:2][::1]
    diff = 'inf'
    if scores[top_2[0]] != 0:
        diff = scores[top_2[1]] - scores[top_2[0]]
    return top_2[0], diff


if __name__ == "__main__":
    from image_viewer import ImageViewer
    from data_test.standard_samples import RANDOM_PAINTING, PAINTINGS_DB, TEST_RETRIEVAL
    from painting_rectification import four_point_transform, remove_pad
    from painting_detection import painting_detection
    dataset_images = [cv2.imread(url) for url in PAINTINGS_DB]

    for painting_path in TEST_RETRIEVAL:
        img = cv2.imread(painting_path)
        painting_contours = painting_detection(img)
        iv = ImageViewer(cols=4)
        for i, corners in enumerate(painting_contours):
            img_sec = four_point_transform(img, corners)
            if not img_sec is None:
                scores = retrieve_painting(
                    img_sec, dataset_images, verbose=False, threshold=30, resize_factor=1, mse=False)
                res, diff = best_match(scores)
                target = dataset_images[res]
                iv.add(img_sec, cmap='bgr')
                iv.add(target, cmap='bgr',
                       title=f'[{res}] {scores[res]} diff={diff}')
        iv.add(img, title=painting_path.split('/')[-1], cmap='bgr')
        iv.show()
