# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from kitti_utils import generate_depth_map
from .SingleDataset import SingleDataset
from cityscapesscripts.helpers.labels import labels
from torchvision import transforms

class KITTISemanticDataset(SingleDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTISemanticDataset, self).__init__(*args, **kwargs)
        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.labels = labels
        self.seman_resize = transforms.Resize((self.height, self.width),
                                               interpolation=pil.NEAREST)
        self.mask = None
    def check_depth(self):
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(os.path.join(self.data_path, folder + '.png'))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_K(self, folder):
        K = np.array([[0.58, 0, 0.5, 0],
                      [0, 1.92, 0.5, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        return K

    def get_rescaleFac(self, folder):
        rescale_fac = 1
        return rescale_fac

    def get_seman(self, folder, do_flip):
        comp = folder.split('/')
        comp[1] = 'semantic'
        newfolder = ''
        for entry in comp:
            newfolder = os.path.join(newfolder, entry)
        semantics = self.loader(os.path.join(self.data_path, newfolder + '.png')).resize(self.full_res_shape, resample = pil.NEAREST)
        semantics = np.array(semantics)[:,:,0]
        trainId_semantics = np.zeros_like(semantics)
        processed_pix = 0
        for i in np.unique(semantics):
            selector = semantics == i
            trainId = self.labels[i].trainId
            trainId_semantics[selector] = trainId
            processed_pix = processed_pix + np.sum(selector)
        assert processed_pix == semantics.shape[0] * semantics.shape[1]
        return trainId_semantics

    def check_seman(self):
        return True
