import os
import cv2
import numpy as np
from step_01_mean_shift_seg import _mean_shift_segmentation
from step_02_mask_largest_segment import _mask_largest_segment
from step_03_dilate_invert import _erode_dilate, _invert, _add_padding
from step_04_connected_components import _find_contours
from step_05_find_paintings import _find_possible_contours
from step_06_erosion import  _clean_frames_noise, _mask_from_contour
from step_07_median_filter import _apply_median_filter
from step_08_canny_edge_detection import _apply_edge_detection
from step_09_hough import _hough
from step_10_find_corners import _find_corners
from step_11_highlight_painting import _highlight_paintings, _draw_all_contours
from step_12_b_create_outer_rect import _mask, draw_rect, rect_contour
from image_viewer import show_me, ImageViewer
import random
from data_test.standard_samples import TEST_PAINTINGS

def scale(img, scale_percent):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dsize = (width, height)
    return cv2.resize(img, dsize)

def scale_heght(img, height):
    return scale(img, height/img.shape[0])

def generate_mask(img, pad=1, area_perc=0.93, img_height=360,
                spatial_radius=3, color_radius=35, maximum_pyramid_level=3,
                size=20,
                filter_contours=True
                ):
    orig_shape = (img.shape[1], img.shape[0])
    img = scale_heght(img, img_height)
    out = _mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level)
    out = _mask_largest_segment(out)
    out = _erode_dilate(out, size)
    out = _invert(out)
    out = _add_padding(out, pad)
    contours = _find_contours(out)
    if filter_contours:
        contours = _find_possible_contours(out, contours)
    paintings_contours = []
    canvas = np.zeros_like(out)
    for contour in contours:
        for_out = _mask_from_contour(out, contour)
        for_out = _clean_frames_noise(for_out)
        for_out = _apply_median_filter(for_out)
        # for_out = _apply_edge_detection(for_out)
        canvas += for_out
    return cv2.resize(canvas[pad:-pad, pad:-pad], orig_shape)

if __name__ == "__main__":
    video_path = 'dataset/photos'
    imgs_path = 'Pytorch-UNet-master/data_paint/imgs'
    masks_path = 'Pytorch-UNet-master/data_paint/masks'
    all_imgs = []

    for root, directories, filenames in os.walk(video_path):
        for filename in filenames: 
            all_imgs.append(os.path.join(root,filename))

    # random.shuffle(all_imgs)

    for i, img_path in enumerate(TEST_PAINTINGS):
        name = img_path.split('\\')[-1]
        img = cv2.imread(img_path)
        mask = generate_mask(img)
        # iv = ImageViewer()
        # iv.add(img, cmap='bgr')
        # iv.add(mask, cmap='bgr')
        # iv.show()
        cv2.imwrite(os.path.join('.', name.replace('.jpg', '.png')), mask)
        # cv2.imwrite(os.path.join(masks_path, name.replace('.jpg', '.png')), mask)
        # cv2.imwrite(os.path.join(imgs_path, name), img)
        l = 128
        print(f'{i:04d} ' + '#'*int(l * (i + 1)/len(all_imgs))+'-'*int(l * (1 - (i + 1)/len(all_imgs))) + f' {(i + 1)*100/len(all_imgs):.0f}%', end='\r')

