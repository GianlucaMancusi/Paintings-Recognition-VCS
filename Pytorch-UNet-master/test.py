from skimage import io, transform
import os

dir_path = '/homes/vpippi/datasets/statue/masks'
print('Start loading...')
all_imgs = [os.path.join(dir_path, f).replace('\\', '/') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
print(f'Loaded {len(all_imgs)} files')
fails = []
for n, i in enumerate(all_imgs):
    print(f'{n}\tTry to read "{i}"   ', end='')
    try:
        io.imread(i)
        print(f'READ')
    except:
        fails.append(i)
        print('FAILED <-----------------------------------')
print(f'fails={fails}')