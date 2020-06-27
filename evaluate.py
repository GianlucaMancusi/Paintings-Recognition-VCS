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

def recall(paint, mask, smooth=1):
    tp = (paint * mask).sum()
    fn = (mask * (255 - paint)).sum()
    return (tp + smooth) / (tp + fn + smooth)

def presicion(paint, mask, smooth=1):
    tp = (paint * mask).sum()
    fp = ((255 - mask) * paint).sum()
    return (tp + smooth) / (tp + fp + smooth)

def specificity(paint, mask, smooth=1):
    tp = (paint * mask).sum()
    tn = (paint * (255 - mask)).sum()
    fn = (mask * (255 - paint)).sum()
    fp = ((255 - mask) * paint).sum()
    return (tn + smooth) / (tn + fp + smooth)

def dice(paint, mask, smooth=1):
    sum_tot = np.sum(paint) + np.sum(mask)
    correct = np.sum(paint[mask > 0])
    fail = np.sum(paint[mask == 0])
    fail = min(correct, fail)
    return (correct - fail) * 2.0 / (sum_tot + smooth)

def tversky(beta, smooth=1):
    assert 0 <= beta <= 1
    alpha = 1 - beta
    def _tversky(paint, mask):
        sum_tot = np.sum(paint) + np.sum(mask)
        tp = (paint * mask).sum()
        fp = ((255 - mask) * paint).sum()
        fn = (mask * (255 - paint)).sum()
        return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return _tversky

def checkup(beta):
    tversky_func = tversky(beta)
    def _checkup(paint, mask):
        dice_val = dice(paint, mask)
        tversky_val = tversky_func(paint, mask)
        specificity_val = specificity(paint, mask)
        presicion_val = presicion(paint, mask)
        recall_val = recall(paint, mask)
        return dice_val, tversky_val, specificity_val, presicion_val, recall_val
    return _checkup


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

def mean(vals):
    return sum(vals) / len(vals)

def evaluate(function, eval_func, test_perc=1.0, seed=0, **kwargs):
    masks_path = 'data_test/paintings_gt/masks'
    imgs_path = 'data_test/paintings_gt/imgs'
    random.seed(seed)

    all_imgs = all_files_in(imgs_path)
    all_masks = all_files_in(masks_path)

    dice_vals = []
    tversky_vals = []
    specificity_vals = []
    presicion_vals = []
    recall_vals = []
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
        # out = np.zeros_like(mask) + 255
        dice, tversky, specificity, presicion, recall = eval_func(out, mask)
        dice_vals.append(dice)
        tversky_vals.append(tversky)
        specificity_vals.append(specificity)
        presicion_vals.append(presicion)
        recall_vals.append(recall)
        print(f'  [{i + 1}/{len(pairs)}] dice={mean(dice_vals):0.4f} tversky={mean(tversky_vals):0.4f} specificity={mean(specificity_vals):0.4f} presicion={mean(presicion_vals):0.4f} recall={mean(recall_vals):0.4f} time={watch.total():.0f}s of "{filename}"', end='\r')
    sys.stdout.write("\033[K")
    time = watch.total()
    return dice_vals, filenames, mean(dice_vals), mean(tversky_vals), mean(specificity_vals), mean(presicion_vals), mean(recall_vals), time

if __name__ == "__main__":
    from data_test.standard_samples import TEST_PAINTINGS
    from image_viewer import ImageViewer
    eval_values = []
    eval_mean = []
    eval_time = []

    test_args = [
        # {'spatial_radius':5, 'color_radius':5, 'maximum_pyramid_level':3},
        # {'size':1,},
        # {'pad':1,},
        # {'pad':10,},
        # {'pad':50,},
        # {'pad':100,},
        # {'filter_contours':True,},
        # {'filter_contours':False,},
        {'beta':0.000000000001,},
        {'beta':0.1,},
        {'beta':0.15,},
        {'beta':0.25,},
        {'beta':0.33,},
        {'beta':0.50,},
        {'beta':0.66,},
        {'beta':0.75,},
        {'beta':0.85,},
        {'beta':0.9,},
    ]

    test_perc = 0.1
    beta = 0.000000000001
    for kwargs in test_args:
        if 'beta' in kwargs:
            beta = kwargs['beta']
            del kwargs['beta']
        dice_vals, filenames, dice_vals, tversky_vals, specificity_vals, presicion_vals, recall_vals, time = evaluate(generate_mask, eval_func=checkup(beta=beta), test_perc=test_perc, **kwargs)
        eval_values.append(dice_vals)
        eval_mean.append(mean)
        eval_time.append(time)
        if test_perc > 0.7:
            with open(f'evaluations/evaluation_{datetime.now().strftime("%m_%d_%H_%M_%S")}.json', 'w') as f:  # writing JSON object
                json.dump({
                    'dice_vals': dice_vals,
                    'filenames': filenames,
                    'dice': dice_vals,
                    'tversky': tversky_vals,
                    'specificity': specificity_vals,
                    'presicion': presicion_vals,
                    'recall': recall_vals,
                    'time': time,
                    'kwargs': kwargs,
                    'test_perc': test_perc,
                }, f)
        info = f'dice={dice_vals:0.04f} tversky={tversky_vals:0.04f} specificity={specificity_vals:0.04f} presicion={presicion_vals:0.04f} recall={recall_vals:0.04f}time={time:.02f}s kwargs={kwargs} beta={beta}{" " * 64}'
        print(f'{info}')

    # for test_img in TEST_PAINTINGS[::-1]:
    #     # test_img = TEST_PAINTINGS[3]
    #     iv = ImageViewer(cols=2)
    #     paint = cv2.imread(test_img)
    #     iv.add(paint, title='orig', cmap='bgr')
    #     for kwargs in test_args:
    #         out = generate_mask(paint, **kwargs)
    #         iv.add(out, title=f'{kwargs}', cmap='bgr')
    #     iv.show()