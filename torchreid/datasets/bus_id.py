from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import re
import sys
import os.path as osp
import numpy as np
from scipy.misc import imsave
import pickle

from torchreid.utils.iotools import mkdir_if_missing, write_json, read_json
from .bases import BaseImageDataset


class BusID(BaseImageDataset):
    """
    BusID

    Dataset statistics:
    # identities: 11131
    # images: 571322
    # cameras: 2
    """
    dataset_dir = 'F:/Database/od_dataset/train_format'

    def __init__(self, root='data', pickle_file="train1-6", split_id=0, verbose=True, **kwargs):
        super(BusID, self).__init__(root)
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.pickle_path = osp.join(r"G:\data_format_transform\new", f'{pickle_file}.pkl')
        self.split_path = osp.join(self.dataset_dir, 'splits.json')

        self._check_before_run()

        self._prepare_split()
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError(
                "split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits) - 1))
        split = splits[split_id]

        train = split['train']
        query = split['query']
        gallery = split['gallery']

        train = [tuple(item) for item in train]
        query = [tuple(item) for item in query]
        gallery = [tuple(item) for item in gallery]

        if verbose:
            print("=> Bus_id loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _prepare_split(self):
        """
        Image name format: 0001001.png, where first four digits represent identity
        and last four digits represent cameras. Camera 1&2 are considered the same
        view and camera 3&4 are considered the same view.
        """
        if not osp.exists(self.split_path):
            print("Creating 10 random splits of train ids and test ids")
            annotations_file = open(self.pickle_path, "rb")
            annotations = pickle.load(annotations_file)
            annotations_file.close()
            img_list = []
            pid_container = set()
            for annotation in annotations:
                image_path = osp.join(self.dataset_dir, annotation["image"]["information"]["path"])
                pid = annotation["annotation"]["persons"][0]["person_id"]
                cam_id = annotation["annotation"]["persons"][0]["direction"]
                box = annotation["annotation"]["persons"][0]["box"]
                img_list.append(([image_path, box], pid, cam_id))
                pid_container.add(pid)

            num_pids = len(pid_container)
            num_train_pids = 7 * num_pids // 8

            splits = []
            for _ in range(10):
                order = np.arange(num_pids)
                np.random.shuffle(order)
                train_idxs = order[:num_train_pids]
                train_idxs = np.sort(train_idxs)
                idx2label = {idx: label for label, idx in enumerate(train_idxs)}

                train, test_a, test_b = [], [], []

                for img_path, pid, camid in img_list:
                    if pid in train_idxs:
                        train.append((img_path, idx2label[pid], camid))
                    else:
                        if camid == 0:
                            test_a.append((img_path, pid, camid))
                        else:
                            test_b.append((img_path, pid, camid))

                # use cameraA as query and cameraB as gallery
                split = {'train': train, 'query': test_a, 'gallery': test_b,
                         'num_train_pids': num_train_pids,
                         'num_query_pids': num_pids - num_train_pids,
                         'num_gallery_pids': num_pids - num_train_pids,
                         }
                splits.append(split)

                # use cameraB as query and cameraA as gallery
                split = {'train': train, 'query': test_b, 'gallery': test_a,
                         'num_train_pids': num_train_pids,
                         'num_query_pids': num_pids - num_train_pids,
                         'num_gallery_pids': num_pids - num_train_pids,
                         }
                splits.append(split)

            print("Totally {} splits are created".format(len(splits)))
            write_json(splits, self.split_path)
            print("Split file saved to {}".format(self.split_path))

        print("Splits created")
