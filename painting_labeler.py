from painting_retrieval import *
from painting_detection import *
from painting_rectification import *
from step_07_highlight_painting import highlight_paintings
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
    def __init__(self, dataset: list, metadata_repository: str, image=None, image_url=None, labels_overflow_limit=20, video_mode=False):
        super().__init__()
        self.dataset = dataset
        self.metadata_repository = InfoTable(metadata_repository)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.labels_overflow_limit = labels_overflow_limit
        self.video_mode = video_mode

    def transform(self, return_info=False, image=None, image_url=None):
        self.image_url = image_url
        self.image = image if self.image_url == None else cv2.imread(image_url)

        if self.image is None or self.dataset is None or self.metadata_repository is None:
            return None

        from step_07_highlight_painting import _draw_all_contours
        painting_contours = painting_detection(self.image)
        out = _draw_all_contours(painting_contours, self.image)

        infos = []
        corners_list = []
        for i, corners in enumerate(painting_contours):
            try:
                y1, y2, x1, x2 = corners[1][0], corners[0][0], corners[2][0], corners[3][0]

                x_low = min(x1[0], y1[0])
                x_high = max(x2[0], y2[0])
                y_low = min(x1[1], x2[1])
                y_high = max(y1[1], y2[1])

                corners_list.append((x_low, x_high, y_low, y_high))

                image_copy = self.image.copy()
                img_sec = four_point_transform(image_copy, corners)
                scores = retrieve_painting(img_sec, self.dataset)
                res, diff = best_match(scores)
                threshold = 400 if self.video_mode else 40

                if diff > threshold:
                    info = self.metadata_repository.painting_info(
                        np.argmin(scores))
                    infos.append(info)
                    title = _resolve_overflow_title_on_word(
                        info["Title"], self.labels_overflow_limit)
                    if title != info["Title"]:
                        title += "..."

                    title = title.replace("à", "a'")
                    title = title.replace("ò", "o'")
                    title = title.replace("è", "e'")
                    title = title.replace("é", "e'")
                    title = title.replace("ù", "u'")

                    font = cv2.FONT_HERSHEY_PLAIN
                    # Since 2 for both fontScale and thickness looked good on FHD images, they're resized according to that proportion
                    fontScale = int((2 * self.image.shape[1]) / 1920)
                    thickness = int((2 * self.image.shape[1]) / 1920)

                    t_size = cv2.getTextSize(
                        title, font, fontScale, thickness)[0]
                    shift = int(self.image.shape[0] * 0.05)
                    left_up_x = min([corners[i][0][0] for i in range(4)])
                    left_up_y = min([corners[i][0][1] for i in range(4)])

                    cv2.rectangle(out, (left_up_x, left_up_y + t_size[1]), (left_up_x + t_size[0], left_up_y), (255, 0, 0), 1)
                    cv2.rectangle(out, (left_up_x, left_up_y + t_size[1]), (left_up_x + t_size[0], left_up_y), (255, 0, 0), -1)
                    cv2.putText(out, title, (left_up_x, left_up_y + t_size[1]), font, fontScale, (255, 255, 255), thickness)
            except Exception as e:
                continue

        return out if not return_info else out, infos, corners_list


if __name__ == "__main__":
    filename = random.choice(TEST_PAINTINGS)

    labeler = PaintingLabeler(dataset=[cv2.imread(
        url) for url in PAINTINGS_DB], metadata_repository='dataset/data.csv')

    iv = ImageViewer(cols=3)

    out = labeler.transform(image_url=filename)
    cv2.imshow("", cv2.resize(out[0], (1920, 1080)))
    cv2.waitKey(0)
