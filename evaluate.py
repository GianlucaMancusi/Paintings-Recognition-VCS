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
import json
from datetime import datetime
import sys

def calc_dice(paint, mask):
    sum_tot = np.sum(paint) + np.sum(mask)
    if sum_tot == 0:
        return 1
    else:
        return np.sum(paint[mask > 0]) * 2.0 / sum_tot

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
    dice_tot = 0
    filenames = []

    if test_perc == 1.0:
        pairs = list(zip(all_imgs, all_masks))
    else:
        pairs = random.sample(list(zip(all_imgs, all_masks)), int(len(all_imgs) * test_perc))

    watch = Stopwatch()
    for i, (paint_path, mask_path) in enumerate(pairs):
        sys.stdout.write("\033[K")
        filename = os.path.basename(paint_path)
        mask = xml2img(mask_path)
        if mask is None:
            print(f'  [{i + 1}/{len(pairs)}] ERROR "{filename}"', end='\r')
            continue
        paint = cv2.imread(paint_path)
        out = function(paint, **kwargs)
        dice = calc_dice(out, mask)
        dice_tot += dice
        dice_vals.append(dice)
        filenames.append(filename)
        print(f'  [{i + 1}/{len(pairs)}] dice_mean={dice_tot/len(dice_vals):0.4f} time={watch.total():.0f}s dice={dice:0.4f} of "{filename}"', end='\r')
    sys.stdout.write("\033[K")
    time = watch.total()
    mean = sum(dice_vals) / len(dice_vals)
    return dice_vals, filenames, mean, time

if __name__ == "__main__":
    eval_values = []
    eval_mean = []
    eval_time = []

    test_args = [
        {},
    ]

    test_perc = 1
    for kwargs in test_args:
        dice_vals, filenames, mean, time = evaluate(generate_mask, test_perc=test_perc, **kwargs)
        eval_values.append(dice_vals)
        eval_mean.append(mean)
        eval_time.append(time)
        with open(f'evaluations/evaluation_{datetime.now().strftime("%m_%d_%H_%M_%S")}.json', 'w') as f:  # writing JSON object
            json.dump({
                'dice_vals': dice_vals,
                'filenames': filenames,
                'mean': mean,
                'time': time,
                'kwargs': kwargs,
                'test_perc': test_perc,
            }, f)
        info = f'mean={mean:0.04f} time={time:.02f}s kwargs={kwargs}'
        print(f'{info}')