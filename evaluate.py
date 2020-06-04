from data_test.standard_samples import get_random_paintings, ALL_PAINTINGS, all_files_in
from shutil import copyfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from generate_painting_masks import generate_mask
from stopwatch import Stopwatch
import multiprocessing as mp
import xmltodict
import random

def calc_dice(paint, mask):
    return np.sum(paint[mask > 0]) * 2.0 / (np.sum(paint) + np.sum(mask))

def xml2img(filename):
    with open(filename) as fd:
        doc = xmltodict.parse(fd.read())

    def get_points(obj):
        pts = []
        if 'polygon' in obj:
            polygon = obj['polygon']
            for pt in polygon['pt']:
                x = pt['x']
                y = pt['y']
                pts.append((x, y))
        return pts
    
    if 'annotation' in doc:
        height = int(doc['annotation']['imagesize']['nrows'])
        width =  int(doc['annotation']['imagesize']['ncols'])

        mask = np.zeros((height, width), dtype=np.uint8)
        if 'object' in doc['annotation']:
            contours = []
            if isinstance(doc['annotation']['object'], list):
                for obj in doc['annotation']['object']:
                    contours.append(get_points(obj))
            else:
                contours.append(get_points(doc['annotation']['object']))
        
            for contour in contours:
                pts = np.array(contour).astype(np.int32)
                mask = cv2.fillPoly(mask, [pts], 255)
        return mask
    return None

def evaluate(function, test_perc=1.0, seed=0, **kwargs):
    masks_path = 'data_test/paintings_gt/masks'
    imgs_path = 'data_test/paintings_gt/imgs'
    random.seed(seed)

    all_imgs = all_files_in(imgs_path)
    all_masks = all_files_in(masks_path)

    dice_vals = []

    if test_perc == 1.0:
        pairs = list(zip(all_imgs, all_masks))
    else:
        pairs = random.sample(list(zip(all_imgs, all_masks)), int(len(all_imgs) * test_perc))

    watch = Stopwatch()
    for i, (paint_path, mask_path) in enumerate(pairs):
        filename = os.path.basename(paint_path)
        mask = xml2img(mask_path)
        if mask is None or mask.max() == 0:
            print(f'[{i + 1}/{len(pairs)}]\tSkipped "{filename}"')
            continue
        paint = cv2.imread(paint_path)
        out = function(paint, **kwargs)
        dice = calc_dice(out, mask)
        dice_vals.append(dice)
        print(f'[{i + 1}/{len(pairs)}] dice={dice:0.4f} of "{filename}"')
    time = watch.stop()
    mean = sum(dice_vals) / len(dice_vals)
    return dice_vals, mean, time

if __name__ == "__main__":
    eval_values = []
    eval_mean = []
    eval_time = []

    test_args = [
        {},
    ]

    for kwargs in test_args:
        dice_vals, mean, time = evaluate(generate_mask, test_perc=0.3, **kwargs)
        eval_values.append(dice_vals)
        eval_mean.append(mean)
        eval_time.append(time)
        print(f'mean={mean:0.04f} time={time:.02f}s kwargs={kwargs}')

    for dice_vals, mean, time in zip(eval_values, eval_mean, eval_time):
        plt.plot(dice_vals, label='dice')
        plt.axhline(y=mean, color='r', ls='--', label='mean')
        plt.title(f'mean: {mean:0.04f}')
        plt.legend()
        plt.show()