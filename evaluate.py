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

tp = 0
tn = 0
fp = 0
fn = 0


def calc_metrics(paint, mask):
    paint[paint < 127] = 0
    paint[paint >= 127] = 255
    mask[mask < 127] = 0
    mask[mask >= 127] = 255

    global tp, tn, fp, fn
    tp = (paint * mask).sum()
    tn = ((255 - paint) * (255 - mask)).sum()
    fn = ((255 - paint) * mask).sum()
    fp = (paint * (255 - mask)).sum()


def recall(smooth=1):
    # Quanti pixel positivi individua la rete 
    return tp / (tp + fn + smooth)


def precision(smooth=1):
    # se io dico che hai una malattia la precision mi dice con che probabilità ho ragione
    return tp / (tp + fp + smooth)


def specificity(smooth=1):
    # se io dico che NON hai una malattia la specificity mi dice con che probabilità ho ragione
    return tn / (tn + fp + smooth)


# def IoU(smooth=1):
#     # se io dico che NON hai una malattia la specificity mi dice con che probabilità ho ragione
#     return tp / (tp + fp + fn)


def dice(smooth=1):
    return tp * 2.0 / (2 * tp + fn + fp + smooth)


def tversky(beta, alpha=None):
    assert 0 <= beta <= 1
    alpha = 1 - beta if alpha is None else alpha

    def _tversky(smooth=1):
        return tp / (tp + alpha * fn + beta * fp + smooth)
    return _tversky


def checkup(beta):
    tversky_func = tversky(beta)
    iou_func = tversky(beta=1, alpha=1)

    def _checkup(paint, mask):
        calc_metrics(paint, mask)
        dice_val = dice()
        tversky_val = tversky_func()
        specificity_val = specificity()
        precision_val = precision()
        recall_val = recall()
        iou_val = iou_func()
        return dice_val, tversky_val, specificity_val, precision_val, recall_val, iou_val
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
    iou_vals = []
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
        dice, tversky, specificity, presicion, recall, iou = eval_func(out, mask)
        dice_vals.append(dice)
        tversky_vals.append(tversky)
        specificity_vals.append(specificity)
        presicion_vals.append(presicion)
        recall_vals.append(recall)
        iou_vals.append(iou)
        print(f'  [{i + 1}/{len(pairs)}] dice={mean(dice_vals):0.4f} tversky={mean(tversky_vals):0.4f} specificity={mean(specificity_vals):0.4f} presicion={mean(presicion_vals):0.4f} recall={mean(recall_vals):0.4f} iou={mean(iou_vals):0.4f} time={watch.total():.0f}s of "{filename}"', end='\r')
    sys.stdout.write("\033[K")
    time = watch.total()
    print(' '.join([f'({i}, {v:0.3f})' for i, v in enumerate(dice_vals, 1)]))
    return dice_vals, filenames, mean(dice_vals), mean(tversky_vals), mean(specificity_vals), mean(presicion_vals), mean(recall_vals), mean(iou_vals), time

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
        # {'beta':0.50,},
        # {'beta':0.50, 'color_radius':10},
        # {'beta':0.50, 'color_radius':20},
        # {'beta':0.50, 'color_radius':35},
        # {'beta':0.50, 'size':31},
        # {'beta':0.50, 'size':41},
        # {'beta':0.50, 'size':21},
        {'beta':0.50,},

    ]

    test_perc = 0.13
    beta = 0.000000000001
    for kwargs in test_args:
        if 'beta' in kwargs:
            beta = kwargs['beta']
            del kwargs['beta']
        dice_vals, filenames, dice_vals, tversky_vals, specificity_vals, presicion_vals, recall_vals, iou_vals, time = evaluate(generate_mask, eval_func=checkup(beta=beta), test_perc=test_perc, **kwargs)
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
        info = f'dice={dice_vals:0.04f} tversky={tversky_vals:0.04f} specificity={specificity_vals:0.04f} presicion={presicion_vals:0.04f} recall={recall_vals:0.04f} iou={iou_vals:0.04f} time={time:.02f}s kwargs={kwargs} beta={beta}{" " * 32}'
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