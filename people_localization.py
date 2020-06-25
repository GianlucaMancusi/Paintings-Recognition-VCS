
from painting_labeler import *
from yolo.people_detection import PeopleDetection
import matplotlib.pyplot as plt
import cv2
import numpy as np


class PeopleLocalization:
    def __init__(self, image=None, image_url=None, people_detection=None):
        super().__init__()
        self.people_labeler = PaintingLabeler(
            [cv2.imread(url) for url in PAINTINGS_DB], "dataset/data.csv")
        self.p_detection = PeopleDetection() if people_detection is None else people_detection
        from data_test.standard_samples import ROOMS
        self.rooms_images = [cv2.imread(url) for url in ROOMS]

    def run(self, image=None, image_url=None):
        out, infos = self.people_labeler.transform(return_info=True, image=image, image_url=image_url)
        room_image = self.rooms_images[0]
        paintings_infos = None

        if infos is not None and len(infos) > 0:

            room_infos = [i['Room'] for i in infos]
            paintings_infos = [i['bb'] for i in infos]

            out, _, peoples = self.p_detection.run(out, paintings_infos)

            room_number = int(np.array(room_infos).mean())
            room_image = self.rooms_images[room_number]

        room_image_resized = cv2.resize(room_image, (316, 216))

        out[out.shape[0]-room_image_resized.shape[0]:out.shape[0],
            :97, :] = room_image_resized[:,:97]
        out[out.shape[0]-room_image_resized.shape[0]+117:out.shape[0],
            97:room_image_resized.shape[1], :] = room_image_resized[117:,97:]

        return out


if __name__ == "__main__":
    # filename = "data_test/paintings_retrieval/011_043.jpg"
    # filename = "data_test/paintings_retrieval/094_037.jpg"
    filename = "data_test/fisheye.jpg"
    # filename = "data_test/paintings_retrieval/045_076.jpg"
    # filename = random.choice(TEST_PAINTINGS)

    localizer = PeopleLocalization()

    out = localizer.run(image_url=filename)

    iv = ImageViewer(cols=2)
    iv.add(out, cmap="bgr")
    iv.show()
    cv2.waitKey(0)
