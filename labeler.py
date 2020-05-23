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

if __name__ == "__main__":
    # filename = "data_test/paintings_retrieval/011_043.jpg"
    # filename = "data_test/paintings_retrieval/094_037.jpg"
    filename = "data_test/paintings_retrieval/093_078_077_073_051_020.jpg"
    # filename = "data_test/paintings_retrieval/045_076.jpg"
    # filename = random.choice(TEST_PAINTINGS)
    

    img = cv2.imread(filename)
    pipeline = Pipeline(default=True)
    pipeline.append(Function(highlight_paintings, source=img, pad=100))
    out = pipeline.run(img, debug=True, print_time=True, filename=filename)

    painting_contours = painting_detection(img)
    dataset_images = [cv2.imread(url) for url in PAINTINGS_DB]
    iv = ImageViewer(cols=3)
    table = InfoTable('dataset/data.csv')
    font = cv2.FONT_HERSHEY_SIMPLEX

    infos = []
    for i, corners in enumerate(painting_contours):
        try:
            y1, y2, x1, x2 = corners[1][0], corners[0][0], corners[2][0], corners[3][0]
            

            x_low = min(x1[0], y1[0])
            x_high = max(x2[0], y2[0])
            y_low = min(x1[1], x2[1])
            y_high = max(y1[1], y2[1])
            paint = img[y_low: y_high, x_low: x_high, :]
            image_copy = img.copy()
            img_sec = four_point_transform(image_copy, corners)
            scores = retrieve_painting(img_sec, dataset_images, resize_factor=0.5, mse=True)
            info = table.painting_info(np.argmin(scores))
            infos.append(info)
            
            label_x = x_low
            label_y = y_low if i % 2 == 0 else y_high
            cv2.putText(out, info["Title"], (label_x, label_y), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        except Exception:
            continue
    
    iv.add(out, cmap="bgr")
    iv.show()
    cv2.waitKey(0)
