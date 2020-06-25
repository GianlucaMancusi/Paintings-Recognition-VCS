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


class PaintingLabeler:
    def __init__(self, dataset: list, metadata_repository: str):
        super().__init__()
        self.dataset = dataset
        self.metadata_repository = InfoTable(metadata_repository)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def transform(self, return_info=False, image=None, image_url=None):
        self.image_url = image_url
        self.image = image if image_url == None else cv2.imread(image_url)
        
        if self.image is None or self.dataset is None or self.metadata_repository is None:
            return None
        

        # out = self.detection_pipeline.run(self.image, debug=False,
        #                   print_time=False, filename=self.image_url)
        #painting_contours = painting_detection(self.image)
        # out = self.detection_pipeline.run(self.image, debug=False, print_time=False, filename=self.image_url)
        from step_11_highlight_painting import _draw_all_contours
        painting_contours = painting_detection(self.image)
        out = _draw_all_contours(painting_contours, self.image)

        infos = []
        for i, corners in enumerate(painting_contours):
            try:
                y1, y2, x1, x2 = corners[1][0], corners[0][0], corners[2][0], corners[3][0]

                x_low = min(x1[0], y1[0])
                x_high = max(x2[0], y2[0])
                y_low = min(x1[1], x2[1])
                y_high = max(y1[1], y2[1])
                paint = self.image[y_low: y_high, x_low: x_high, :]
                image_copy = self.image.copy()
                img_sec = four_point_transform(image_copy, corners)
                scores = retrieve_painting(img_sec, self.dataset)
                res, diff = best_match(scores)
                if diff > 50:
                    info = self.metadata_repository.painting_info(np.argmin(scores))
                    info['bb'] = (x_low,x_high, y_low, y_high)
                    infos.append(info)

                    label_x = x_low
                    label_y = y_low if i % 2 == 0 else y_high
                    cv2.putText(out, info["Title"], (label_x, label_y),
                                self.font, 1, (255, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                # print(e)
                continue

        return out if not return_info else out, infos


if __name__ == "__main__":
    # filename = "data_test/paintings_retrieval/011_043.jpg"
    # filename = "data_test/paintings_retrieval/094_037.jpg"
    filename = "data_test/paintings_retrieval/093_078_077_073_051_020.jpg"
    # filename = "data_test/paintings_retrieval/045_076.jpg"
    # filename = random.choice(TEST_PAINTINGS)

    labeler = PaintingLabeler(dataset=[cv2.imread(
        url) for url in PAINTINGS_DB], metadata_repository='dataset/data.csv')

    iv = ImageViewer(cols=3)

    out = labeler.transform(image_url=filename)
    iv.add(out, cmap="bgr")
    iv.show()
    cv2.waitKey(0)
