# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions
import warnings
import torch
from trainer import Trainer
a = torch.zeros([4, 4, 4]).cuda()
print("blabla1")
warnings.filterwarnings("ignore")


options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    a = torch.zeros([4, 4, 4]).cuda()
    print("blabla")
    trainer = Trainer(opts)
    trainer.train()
