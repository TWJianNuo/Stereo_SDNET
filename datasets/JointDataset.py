# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch.utils.data as data

class JointDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 datasets
                 ):
        super(JointDataset, self).__init__()
        self.datasets = datasets
        self.dataset_num = len(datasets)
        self.sample_num = np.zeros(len(self.datasets), dtype=np.int)
        self.dec_bar = np.zeros(len(self.datasets), dtype=np.int)
        self.format = list()
        # self.item_recList = list()
        for idx, sDataset in enumerate(self.datasets):
            self.sample_num[idx] = sDataset.__len__()
            self.format.append((sDataset.tag, sDataset.height, sDataset.width))
            self.dec_bar[idx] = np.sum(self.sample_num[0:idx+1])

    def __len__(self):
        return np.int(np.sum(self.sample_num))

    def __getitem__(self, index):
        bulk_ind = np.digitize(index, self.dec_bar)
        assert bulk_ind < self.dataset_num, "Joint dataset read error"
        spec_ind = np.int(index - np.sum(self.sample_num[0 : bulk_ind]))
        # self.item_recList.append(index)
        return self.datasets[bulk_ind].__getitem__(spec_ind)
