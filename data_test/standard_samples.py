from os import listdir
from os.path import isfile, join
import random

dir_path = 'data_test/paintings'
TEST_PAINTINGS = [join(dir_path, f).replace('\\', '/') for f in listdir(dir_path) if isfile(join(dir_path, f))]

dir_path = 'data_test/paintings_retrieval'
TEST_RETRIEVAL = [join(dir_path, f).replace('\\', '/') for f in listdir(dir_path) if isfile(join(dir_path, f))]

dir_path = 'dataset/paintings_db'
PAINTINGS_DB = [join(dir_path, f).replace('\\', '/') for f in listdir(dir_path) if isfile(join(dir_path, f))]

FISH_EYE = 'data_test/fisheye.jpg'
PEOPLE = 'data_test/persone.jpg'

RANDOM_PAINTING = random.choice(TEST_PAINTINGS)

if __name__ == "__main__":
    print(TEST_PAINTINGS)
    import pathlib
    pathlib.Path().absolute()