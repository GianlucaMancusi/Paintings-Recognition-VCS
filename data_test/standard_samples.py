import os
import random

dir_path = 'data_test/paintings'
TEST_PAINTINGS = [os.path.join(dir_path, f).replace('\\', '/') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

dir_path = 'data_test/paintings_retrieval'
TEST_RETRIEVAL = [os.path.join(dir_path, f).replace('\\', '/') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

dir_path = 'data_test/camera_correction'
TEST_DISTORTION = [os.path.join(dir_path, f).replace('\\', '/') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

dir_path = 'dataset/paintings_db'
PAINTINGS_DB = [os.path.join(dir_path, f).replace('\\', '/') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

try:
    path = 'dataset/photos'
    ALL_PAINTINGS = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.jpg']
except:
    pass

FISH_EYE = 'data_test/fisheye.jpg'
CHESSBOARD = 'data_test/chessboard.png'
PEOPLE = 'data_test/persone.jpg'
PERSPECTIVE = 'data_test/perspective.png'

RANDOM_PAINTING = random.choice(TEST_PAINTINGS)

def get_random_paintings(number):
    return random.sample(ALL_PAINTINGS, number)

def all_files_in(dir_path):
    return [os.path.join(dir_path, f).replace('\\', '/') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

if __name__ == "__main__":
    # print(TEST_PAINTINGS)
    # import pathlib
    # pathlib.Path().absolute()
    print(get_random_paintings(10))