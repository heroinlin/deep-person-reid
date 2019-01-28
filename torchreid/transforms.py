from __future__ import absolute_import
from __future__ import division

import torch

from torchvision import transforms as tv_transforms
import cv2
import random
import numpy as np


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5):
        self.height = height
        self.width = width
        self.p = p

    def __call__(self, img):
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = cv2.resize(img, dsize=(new_width, new_height))
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img[x1: x1 + self.width, y1: y1 + self.height, :]
        return croped_img


def build_transforms(height, width, is_train, **kwargs):
    """Build transforms

    Args:
    - height (int): target image height.
    - width (int): target image width.
    - is_train (bool): train or test phase.
    """
    
    # use imagenet mean and std as default
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    transforms = []

    if is_train:
        transforms += [Random2DTranslation(height, width)]
        # transforms += [tv_transforms.RandomHorizontalFlip()]
        # else:
        #     transforms += [tv_transforms.Resize((height, width))]

    transforms += [tv_transforms.ToTensor()]
    transforms += [tv_transforms.Normalize(imagenet_mean, imagenet_std)]
    transforms = tv_transforms.Compose(transforms)

    return transforms