from predict import *
from utils.xml2img import xml2img
from os import listdir, walk
from os.path import isfile, join, splitext
from skimage import io
from PIL import Image

tp = 0
tn = 0
fp = 0
fn = 0


def calc_metrics(paint, mask):
    paint[paint < 127] = 0
    paint[paint >= 127] = 255
    mask[mask < 127] = 0
    mask[mask >= 127] = 255

    if len(mask.shape) == 2:
        mask = np.stack([mask, mask, mask], axis=2)
    global tp, tn, fp, fn
    tp = (paint * mask).sum()
    tn = ((255 - paint) * (255 - mask)).sum()
    fn = ((255 - paint) * mask).sum()
    fp = (paint * (255 - mask)).sum()


def recall(smooth=1):
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


def checkup(beta, alpha=None):
    tversky_func = tversky(beta, alpha)
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


def mean(values):
    return sum(values) / len(values)


# dir_img_val = '/homes/vpippi/datasets/paintings_gt/imgs/'
# dir_mask_val = '/homes/vpippi/datasets/paintings_gt/masks/'
# dir_img_val = '/homes/vpippi/datasets/paintings/imgs/'
# dir_mask_val = '/homes/vpippi/datasets/statue/masks/'
dir_img_val = '/homes/vpippi/datasets/statue_gt/imgs/'
dir_mask_val = '/homes/vpippi/datasets/statue_gt/masks/'


def evaluate(net, beta, alpha, device='cuda'):
    net.eval()
    images_fn = [join(dir_img_val, f).replace('\\', '/') for f in listdir(dir_img_val) if isfile(join(dir_img_val, f))]
    masks_fn = [join(dir_mask_val, f).replace('\\', '/') for f in listdir(dir_mask_val) if
                isfile(join(dir_mask_val, f))]
    # assert len(images_fn) == len(masks_fn)
    images_name = [name.split('/')[-1][:-4] for name in images_fn]
    # masks_name = [name.split('/')[-1][:-4] for name in masks_fn]
    masks_fn = [m for m in masks_fn if m.split('/')[-1][:-4] in images_name]
    images_fn.sort()
    masks_fn.sort()
    check = checkup(beta, alpha)
    dice_vals = []
    tversky_vals = []
    specificity_vals = []
    precision_vals = []
    recall_vals = []
    iou_vals = []
    for i, (img_fn, mask_fn) in enumerate(zip(images_fn, masks_fn), 1):
        if img_fn.split('/')[-1][:-4] != mask_fn.split('/')[-1][:-4]:
            continue
        img = Image.open(img_fn)
        mask = np.array(Image.open(mask_fn))
        if len(mask.shape) == 2:
            mask = np.stack([mask, mask, mask], axis=2)
        # mask = xml2img(mask_fn)
        # if mask.sum() == 0:
        #     continue
        mask_pred = predict_img(net=net, full_img=img, out_threshold=0.5, device=device)
        mask_pred = mask_pred.astype(np.uint8) * 255
        dice_val, tversky_val, specificity_val, precision_val, recall_val, iou_val = check(mask, mask_pred)
        dice_vals.append(dice_val)
        tversky_vals.append(tversky_val)
        specificity_vals.append(specificity_val)
        precision_vals.append(precision_val)
        recall_vals.append(recall_val)
        iou_vals.append(iou_val)
        # print(i, mask.sum(), dice_val, tversky_val, specificity_val, precision_val, recall_val, iou_val)
    return mean(dice_vals), mean(tversky_vals), mean(specificity_vals), mean(precision_vals), mean(recall_vals), mean(iou_vals)


if __name__ == '__main__':
    args = get_args()
    print(f'Evaluation starting...')
    net = UNet(n_channels=3, n_classes=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    print(f'Model loaded "{args.model}"')
    evaluate(net, args.beta, args.alpha, device)
