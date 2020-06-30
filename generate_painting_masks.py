import os
import cv2
import numpy as np
from step_01_pre_processing import _mean_shift_segmentation
from step_02_background_detection import _mask_largest_segment
from step_03_cleaning import _closing, _invert, _add_padding
from step_04_components_selection import _find_contours
from step_04_components_selection import _find_possible_contours
from step_05_contour_pre_processing import  _clean_frames_noise, _mask_from_contour
from step_05_contour_pre_processing import _apply_median_filter
from step_05_contour_pre_processing import _apply_edge_detection
from step_06_corners_detection import _hough
from step_06_corners_detection import _find_corners
from step_07_highlight_painting import _highlight_paintings, _draw_all_contours
from step_08_b_create_outer_rect import _mask, draw_rect, rect_contour
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

def generate_mask(img, pad=1, area_perc=0.93, img_height=720,
                spatial_radius=5, color_radius=10, maximum_pyramid_level=3,
                size=21,
                filter_contours=True
                ):
    orig_shape = (img.shape[1], img.shape[0])
    img = scale_heght(img, img_height)
    out = _mean_shift_segmentation(img, spatial_radius, color_radius, maximum_pyramid_level)
    out = _mask_largest_segment(out)
    out = _closing(out, size)
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
        canvas += for_out
    return cv2.resize(canvas[pad:-pad, pad:-pad], orig_shape)

if __name__ == "__main__":
    video_path = 'dataset/photos'
    imgs_path = 'Pytorch-UNet-master/data_statue/imgs'
    masks_path = 'Pytorch-UNet-master/data_statue/masks'
    all_imgs = []

    for root, directories, filenames in os.walk(video_path):
        for filename in filenames: 
            all_imgs.append(os.path.join(root,filename))

    all_imgs = [ s.replace(s.split('\\')[-1], s.split('\\')[-1].lower()) for s in all_imgs]

    for i, img_path in enumerate(all_imgs):
        name = img_path.split('\\')[-1]
        img = cv2.imread(img_path)
        mask = np.zeros_like(img)
        cv2.imwrite(os.path.join(imgs_path, name), img)
        cv2.imwrite(os.path.join(masks_path, name.replace('.jpg', '.png')), mask)
        print(f'  image [{i}/{len(all_imgs)}]', end='\r')
