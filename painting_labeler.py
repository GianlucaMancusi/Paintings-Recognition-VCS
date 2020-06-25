from painting_retrieval import *
from painting_detection import *
from painting_rectification import *
from step_11_highlight_painting import highlight_paintings
from pipeline import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
from data_test.standard_samples import RANDOM_PAINTING, PAINTINGS_DB, TEST_RETRIEVAL, TEST_PAINTINGS
from csv_reader import InfoTable
import random


def _resolve_overflow_title_on_word(s: str, max_letters: int):
    words = s.split(" ")
    letters_count = 0
    result = ""
    for w in words:
        if len(w) + len(result) > max_letters:
            return result
        
        result += (w + " ")
    
    return result[:-1] if len(result) > 0 else s


class PaintingLabeler:
    def __init__(self, dataset: list, metadata_repository: str, image=None, image_url=None, labels_overflow_limit=50):
        super().__init__()
        assert(image is not None or image_url is not None)
        self.image_url = image_url
        self.image = image if image_url == None else cv2.imread(image_url)
        self.dataset = dataset
        self.metadata_repository = InfoTable(metadata_repository)

        self.detection_pipeline = Pipeline(default=True)
        self.detection_pipeline.append(
            Function(highlight_paintings, source=self.image, pad=100))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.labels_overflow_limit = labels_overflow_limit

    def fit(self, image_url: str, dataset: list, metadata_repository: str):
        self.image_url = image_url
        self.image = cv2.imread(image_url)
        self.dataset = dataset
        self.metadata_repository = InfoTable(metadata_repository)

    def transform(self, return_info=False):
        if self.image is None or self.dataset is None or self.metadata_repository is None:
            return None

        from step_11_highlight_painting import _draw_all_contours
        painting_contours = painting_detection(self.image)
        out = _draw_all_contours(painting_contours, self.image)

        infos = []
        for i, corners in enumerate(painting_contours):
            try:
                y1, y2, x1, x2 = corners[1][0], corners[0][0], corners[2][0], corners[3][0]


                image_copy = self.image.copy()
                img_sec = four_point_transform(image_copy, corners)
                scores = retrieve_painting(img_sec, self.dataset)
                res, diff = best_match(scores)
                if diff > 50:
                    info = self.metadata_repository.painting_info(
                        np.argmin(scores))
                    infos.append(info)

                    title = _resolve_overflow_title_on_word(info["Title"], self.labels_overflow_limit)
                    if title != info["Title"]:
                        title += "..."
                    
                    font = cv2.FONT_HERSHEY_PLAIN
                    # Since 2 for both fontScale and thickness looked good on FHD images, they're resized according to that proportion
                    fontScale = int((2 * self.image.shape[1]) / 1920)
                    thickness = int((2 * self.image.shape[1]) / 1920)

                    t_size = cv2.getTextSize(title, font, fontScale, thickness)[0]
                    shift = int(self.image.shape[0] * 0.05)
                    left_up_x = min([corners[i][0][0] for i in range(4)])
                    left_up_y = min([corners[i][0][1] for i in range(4)]) + shift 
                    
                    cv2.rectangle(out, (left_up_x, left_up_y + t_size[1]), (left_up_x + t_size[0], left_up_y), (255, 0, 0), 1)
                    cv2.rectangle(out, (left_up_x, left_up_y + t_size[1]), (left_up_x + t_size[0], left_up_y), (255, 0, 0), -1)
                    cv2.putText(out, title, (left_up_x, left_up_y + t_size[1]), font, fontScale, (255, 255, 255), thickness)
                    
            except AttributeError as e:
                # print(e)
                continue

        return out if not return_info else out, infos

    def fit_transform(self, image_url: str, dataset: list, metadata_repository: str):
        self.fit(image_url=image_url, dataset=dataset,
                 metadata_repository=metadata_repository)
        return self.transform()


if __name__ == "__main__":
    '''# filename = "data_test/paintings_retrieval/011_043.jpg"
    # filename = "data_test/paintings_retrieval/094_037.jpg"
    # filename = "data_test/paintings_retrieval/093_078_077_073_051_020.jpg"
    filename = "data_test/paintings_retrieval/045_076.jpg"
    # filename = "data_test/paintings/8.jpg"
    #filename = random.choice(TEST_PAINTINGS)


    db = [cv2.imread(url) for url in PAINTINGS_DB]
    labeler = PaintingLabeler(image_url=filename, dataset=db, metadata_repository='dataset/data.csv')

    iv = ImageViewer(cols=3)

    out = labeler.transform()
    iv.add(out)
    #for p in TEST_PAINTINGS:
    #    out = labeler.fit_transform(image_url=p, dataset=db, metadata_repository='dataset/data.csv')[0]
    #    cv2.imshow(p, cv2.resize(out, (1920, 1080)))
    iv.show()
    cv2.waitKey(0)
    '''
    
    # filename = "data_test/paintings_retrieval/011_043.jpg"
    # filename = "data_test/paintings_retrieval/094_037.jpg"
    # filename = "data_test/paintings_retrieval/093_078_077_073_051_020.jpg"
    # filename = "data_test/paintings_retrieval/045_076.jpg"
    # filename = "data_test/paintings/8.jpg"
    filename = random.choice(TEST_PAINTINGS)

    labeler = PaintingLabeler(image_url=filename, dataset=[cv2.imread(url) for url in PAINTINGS_DB], metadata_repository='dataset/data.csv')

    iv = ImageViewer(cols=3)

    out = labeler.transform()
    iv.add(out[0], cmap="bgr")
    iv.show()
    cv2.waitKey(0)

