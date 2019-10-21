# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), isMulChannel = False, use_skips=True, isSwitch = False, outputtwoimage = False):
        super(DepthDecoder, self).__init__()

        semanticType = 19
        if isMulChannel:
            self.num_output_channels = semanticType
        else:
            self.num_output_channels = 1
        if outputtwoimage:
            self.num_output_channels = 2
        else:
            self.num_output_channels = 1
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        # self.commonScale = semanticScale
        self.semanticType = 19
        self.isSwitch = isSwitch

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder for depth
        self.convs = OrderedDict()
        if not self.isSwitch:
            for i in range(4, -1, -1):
                # upconv_0
                num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            # For semantics
            # for i in range(4, -1, -1):
            #     # upconv_0
            #     num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            #     num_ch_out = self.num_ch_dec[i]
            #     self.convs[("upconv_seman", i, 0)] = ConvBlock(num_ch_in, num_ch_out)
            #
            #     # upconv_1
            #     num_ch_in = self.num_ch_dec[i]
            #     if self.use_skips and i > 0:
            #         num_ch_in += self.num_ch_enc[i - 1]
            #     num_ch_out = self.num_ch_dec[i]
            #     self.convs[("upconv_seman", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            for s in self.scales:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

            # for s in self.scales:
            #     self.convs[("semanconv", s)] = Conv3x3(self.num_ch_dec[s], self.semanticType)

        else:
            for i in range(4, -1, -1):
                # upconv_0
                num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("switchconv", i, 0)] = SwitchBlock(num_ch_in, num_ch_out)

                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("switchconv", i, 1)] = SwitchBlock(num_ch_in, num_ch_out)

            for s in self.scales:
                self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

            for s in self.scales:
                self.convs[("semanconv", s)] = Conv3x3(self.num_ch_dec[s], self.semanticType)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()
        # self.sfx = nn.Softmax()

    def forward(self, input_features, computeSemantic = False, computeDepth = True):
        self.outputs = {}

        # decoder
        # We confirm that both will be full scale
        if not self.isSwitch:
            x = input_features[-1]
            if computeDepth:
                xdD = dict()
                for i in range(4, -1, -1):
                    x = self.convs[("upconv", i, 0)](x)
                    x = [upsample(x)]
                    if self.use_skips and i > 0:
                        x += [input_features[i - 1]]
                    x = torch.cat(x, 1)
                    x = self.convs[("upconv", i, 1)](x)
                    xdD[i] = x
                for i in self.scales:
                    self.outputs[("mul_disp", i)] = self.sigmoid(self.convs[("dispconv", i)](xdD[i]))

            # y = input_features[-1]
            # if computeSemantic:
            #     xdS = dict()
            #     for i in range(4, -1, -1):
            #         y = self.convs[("upconv_seman", i, 0)](y)
            #         y = [upsample(y)]
            #         if self.use_skips and i > 0:
            #             y += [input_features[i - 1]]
            #         y = torch.cat(y, 1)
            #         y = self.convs[("upconv_seman", i, 1)](y)
            #         xdS[i] = y
            #     for i in self.scales:
            #         self.outputs[("seman", i)] = self.convs[("semanconv", i)](xdS[i])
        else:
            if computeDepth:
                t = input_features[-1]
                tD = dict()
                for i in range(4, -1, -1):
                    t = self.convs[("switchconv", i, 0)](t, False)
                    t = [upsample(t)]
                    if self.use_skips and i > 0:
                        t += [input_features[i - 1]]
                    t = torch.cat(t, 1)
                    t = self.convs[("switchconv", i, 1)](t, False)
                    tD[i] = t
                for i in self.scales:
                    self.outputs[("mul_disp", i)] = self.sigmoid(self.convs[("dispconv", i)](tD[i]))
            if computeSemantic:
                t = input_features[-1]
                tS = dict()
                for i in range(4, -1, -1):
                    t = self.convs[("switchconv", i, 0)](t, True)
                    t = [upsample(t)]
                    if self.use_skips and i > 0:
                        t += [input_features[i - 1]]
                    t = torch.cat(t, 1)
                    t = self.convs[("switchconv", i, 1)](t, True)
                    tS[i] = t
                for i in self.scales:
                    self.outputs[("seman", i)] = self.convs[("semanconv", i)](tS[i])

        return self.outputs
