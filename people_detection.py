from yolo.people_detection import run
import cv2



if __name__ == "__main__":
    from data_test.standard_samples import RANDOM_PAINTING
    img = cv2.imread('data_test\persone.jpg')
    ris, _ = run(img)
    cv2.imshow("Ris", ris)
    cv2.waitKey()