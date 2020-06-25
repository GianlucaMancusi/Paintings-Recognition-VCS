
from painting_labeler import *
from yolo.people_detection import PeopleDetection
import matplotlib.pyplot as plt
import cv2
import numpy as np


class PeopleLocalization:
    def __init__(self, dataset: list, metadata_repository: str, image=None, image_url=None):
        super().__init__()
        self.people_labeler = PaintingLabeler(
            dataset, metadata_repository, image, image_url)
        self.p_detection = PeopleDetection()
        from data_test.standard_samples import ROOMS
        self.rooms_images = [cv2.imread(url) for url in ROOMS]

    def run(self):
        out, infos = self.people_labeler.transform(return_info=True)

        if infos is None or len(infos) == 0:
            return False

        room_infos = [i['Room'] for i in infos]

        room_number = int(np.array(room_infos).mean())

        room_image = self.rooms_images[room_number]

        room_image_resized = cv2.resize(room_image, (316, 216))

        out[:room_image_resized.shape[0],
            :room_image_resized.shape[1], :] = room_image_resized

        ris, _, peoples = self.p_detection.run(out)
        if peoples == 0:
            return False

        return ris


if __name__ == "__main__":
    # filename = "data_test/paintings_retrieval/011_043.jpg"
    # filename = "data_test/paintings_retrieval/094_037.jpg"
    filename = "data_test/persone.jpg"
    # filename = "data_test/paintings_retrieval/045_076.jpg"
    # filename = random.choice(TEST_PAINTINGS)

    localizer = PeopleLocalization(image_url=filename, dataset=[cv2.imread(
        url) for url in PAINTINGS_DB], metadata_repository='dataset/data.csv')

    out = localizer.run()

    iv = ImageViewer(cols=2)
    iv.add(out, cmap="bgr")
    iv.show()
    cv2.waitKey(0)
