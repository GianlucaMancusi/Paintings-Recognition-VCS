from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from skimage import io, transform
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from .xml2img import xml2img
from .canny import CannyFilter


class Resize(transforms.Resize):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = F.resize(image, self.size, self.interpolation)
        mask = F.resize(mask, self.size, self.interpolation)
        return {'image': image, 'mask': mask}


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and image.size[0] < self.size[1]:
            image = F.pad(image, (self.size[1] - image.size[0], 0), self.fill, self.padding_mode)
            mask = F.pad(mask, (self.size[1] - mask.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and image.size[1] < self.size[0]:
            image = F.pad(image, (0, self.size[0] - image.size[1]), self.fill, self.padding_mode)
            mask = F.pad(mask, (0, self.size[0] - mask.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        image = F.crop(image, i, j, h, w)
        mask = F.crop(mask, i, j, h, w)
        return {'image': image, 'mask': mask}


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = F.center_crop(image, self.size)
        mask = F.center_crop(mask, self.size)
        return {'image': image, 'mask': mask}


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return {'image': image, 'mask': mask}


class RandomRotation(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        angle = np.random.choice([0, 90, 180, 270], 1)[0]
        image = F.rotate(image, angle, False, False, None, None)
        mask = F.rotate(mask, angle, False, False, None, None)
        return {'image': image, 'mask': mask}


class ColorJitter(transforms.ColorJitter):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        t = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        return {'image': t(image), 'mask': mask}


class RandomGrayscale(transforms.RandomGrayscale):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        num_output_channels = 1 if image.mode == 'L' else 3
        if np.random.rand() < self.p:
            image = F.to_grayscale(image, num_output_channels=num_output_channels)
        return {'image': image, 'mask': mask}


class RandomPerspective(transforms.RandomPerspective):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        if np.random.rand() < self.p:
            width, height = image.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            image = F.perspective(image, startpoints, endpoints, self.interpolation)
            mask = F.perspective(mask, startpoints, endpoints, self.interpolation)
        return {'image': image, 'mask': mask}


class ToPILImage(transforms.ToPILImage):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': F.to_pil_image(image, self.mode), 'mask': F.to_pil_image(mask, self.mode)}


class ToTensor(transforms.ToTensor):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': F.to_tensor(image), 'mask': F.to_tensor(mask)}


class Canny(object):

    def __init__(self,
                 low_threshold=None,
                 high_threshold=None,
                 hysteresis=False,
                 k_gaussian=3,
                 mu=0,
                 sigma=1,
                 k_sobel=3,
                 device=None):
        self.filter = CannyFilter(k_gaussian=k_gaussian, mu=mu, sigma=sigma, k_sobel=k_sobel, device=device)
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hysteresis = hysteresis

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        with torch.no_grad():
            _, _, _, grad_magnitude, _, _ = self.filter.forward(image.unsqueeze(0),
                                                                self.low_threshold,
                                                                self.high_threshold,
                                                                self.hysteresis
                                                                )
            image = grad_magnitude.squeeze(0)
        return {'image': image, 'mask': mask}


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, transforms_func=None, train=True):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.transforms_func = transforms_func
        if transforms_func is None:
            if train:
                self.transforms_func = transforms.Compose([
                    ToPILImage(),
                    Resize(512),
                    RandomCrop(512),
                    RandomRotation(),
                    RandomHorizontalFlip(),
                    ColorJitter(0.4, 0.4, 0.4, 0.1),
                    RandomGrayscale(),
                    # RandomPerspective(),
                    ToTensor(),
                    # Canny(),
                ])
            else:
                self.transforms_func = transforms.Compose([
                    ToPILImage(),
                    Resize(512),
                    CenterCrop(512),
                    ToTensor(),
                    # Canny(),
                ])
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        tf = transforms.Compose([
            transforms.Resize(512),
            transforms.ToTensor(),
        ])
        return tf(pil_img)

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        if len(mask_file) == 0:
            mask_file = glob(self.masks_dir + idx.lower() + '.*')
        if len(img_file) == 0:
            img_file = glob(self.imgs_dir + idx.lower() + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        # mask = Image.open(mask_file[0])
        # img = Image.open(img_file[0])
        img = io.imread(img_file[0])
        if mask_file[0].endswith('.xml'):
            mask = xml2img(mask_file[0])
        else:
            mask = io.imread(mask_file[0])

        # assert img.size == mask.size, \
        #     f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        if len(mask.shape) == 3:
            mask = np.amax(mask, axis=2)
        sample = {'image': img, 'mask': mask}
        if self.transforms_func:
            sample = self.transforms_func(sample)

        # img = self.preprocess(img, self.scale)
        # mask = self.preprocess(mask, self.scale)

        return sample
