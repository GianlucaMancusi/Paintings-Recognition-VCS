from PIL import Image
import torchvision.transforms.functional as F
import os

dir_path = '/homes/vpippi/datasets/paintings/masks'
all_masks = [os.path.join(dir_path, f).replace('\\', '/') for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

all_white = 0
all_pix = 0

for i, mask in enumerate(all_masks):
    image = Image.open(mask)
    x = F.to_tensor(image)
    white = x.sum().item()
    all = x.numel()
    all_white += white
    all_pix += all
    print(f'[{i+1}/{len(all_masks)}] total={all_white*100/all_pix:.02f}% solo={white*100/all:.02f}%')