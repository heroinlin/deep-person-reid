from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import cv2
import numpy as np
import os.path as osp
import io

import torch
from torch.utils.data import Dataset


def process_box(box, image_width, image_height):
    x1, y1, x2, y2 = box
    x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    # 框扩大1.5倍
    w = min(w * 1.5, 1.0)
    h = min(h * 1.5, 1.0)
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    # 到图像范围
    x1, y1, x2, y2 = round(x1 * image_width), round(y1 * image_height), round(x2 * image_width), round(y2 * image_height)
    x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, image_width), min(y2, image_height)
    box = [x1, y1, x2, y2]
    return box


def read_image(img_path, camid="0"):

    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    image_path = img_path[0]
    box = img_path[1]

    if not osp.exists(image_path):
        raise IOError("{} does not exist".format(image_path))
    while not got_img:
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8()), 1)
            box = process_box(box, img.shape[1], img.shape[0])
            img = img[box[1]:box[3], box[0]:box[2], :]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, dsize=(144, 144))
            if camid:
                img = cv2.flip(img, -1)
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid, img_path[0]


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    _sample_methods = ['evenly', 'random', 'all']

    def __init__(self, dataset, seq_len=15, sample_method='evenly', transform=None):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample_method = sample_method
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)

        if self.sample_method == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        
        elif self.sample_method == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num/self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32)*(num-1)])
            assert len(indices) == self.seq_len
        
        elif self.sample_method == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        
        else:
            raise ValueError('Unknown sample method: {}. Expected one of {}'.format(self.sample_method, self._sample_methods))

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid