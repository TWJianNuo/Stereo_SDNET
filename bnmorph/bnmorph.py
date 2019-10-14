import math
import matplotlib.pyplot as plt
import PIL.Image as pil
from torch import nn
from torch.autograd import Function
from utils import *
from numba import jit
import torch

import bnmorph_getcorpts

torch.manual_seed(42)

@jit(nopython=True)
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

@jit(nopython=True)
def window_function(x, smooth_range):
    sigmoid_bd = 12 / smooth_range
    bias_pos = smooth_range / 2 + 1
    bias_neg = 0 - smooth_range / 2
    if x < 1 and x > 0:
        y = 1
    elif x <= 0:
        y = sigmoid_function(sigmoid_bd * (x - bias_neg))
    else:
        y = sigmoid_function(-sigmoid_bd * (x - bias_pos))
    return y

@jit(nopython=True)
def distance_function(x, pixel_range):
    sigmoid_bd = 12 / pixel_range
    bias = pixel_range / 2
    y = sigmoid_function(-sigmoid_bd * (x - bias))
    return y
@jit(nopython=True)
def morph_along_lines(height, width, morphed_x, morphed_y, recorder, srcx_set, srcy_set, dstx_set, dsty_set, fig = 10):
    ratio = 1
    smooth_range = 0.5
    pixel_range = 40
    alpha_padding = 0.1
    for dimy in range(height):
        for dimx in range(width):
            for k in range(srcy_set.shape[0]):
                srcx = srcx_set[k]
                srcy = srcy_set[k]
                dstx = dstx_set[k]
                dsty = dsty_set[k]

                dragged_srcx = dstx + (ratio+1) * (srcx - dstx)
                dragged_srcy = dsty + (ratio+1) * (srcy - dsty)

                alpha = ((dimx - dragged_srcx) * (dstx - dragged_srcx) + (dimy - dragged_srcy) * (dsty - dragged_srcy)) / ((dragged_srcx - dstx)*(dragged_srcx - dstx) + (dragged_srcy - dsty)*(dragged_srcy - dsty) + 1e-5)

                d2src = np.sqrt((dragged_srcx - dimx)**2 + (dragged_srcx - dimy)**2)
                d2dst = np.sqrt((dstx - dimx)**2 + (dsty - dimy)**2)
                d2line = np.abs((dstx - srcx) * (srcy - dimy) - (srcx - dimx) * (dsty - srcy)) / (np.sqrt( (srcx - dstx)*(srcx - dstx) + (srcy - dsty)*(srcy - dsty) ) + 1e-5)

                if alpha < 1 and alpha > 0:
                    dgeneral = d2line
                else:
                    if d2src > d2dst:
                        dgeneral = d2dst
                    else:
                        dgeneral = d2src

                alpha_weight = window_function(alpha, smooth_range)
                pixel_range_weight = distance_function(dgeneral, pixel_range)
                recorder[k, 0] = distance_function(dgeneral, 20)

                recorder[k,1] = pixel_range_weight * alpha_weight * (dragged_srcx - dstx) * alpha * 0.8
                recorder[k,2] = pixel_range_weight * alpha_weight * (dragged_srcy - dsty) * alpha * 0.8

            totweights = 0
            avex = 0
            avey = 0
            for k in range(srcy_set.shape[0]):
                totweights = totweights + recorder[k][0]
            for k in range(srcy_set.shape[0]):
                avex = avex + recorder[k,0] / (totweights + 1e-4) * recorder[k,1]
                avey = avey + recorder[k,0] / (totweights + 1e-4) * recorder[k,2]
            morphed_x[dimy][dimx] = avex + dimx
            morphed_y[dimy][dimx] = avey + dimy

    """
    ratio = 1
    smooth_range = 1
    pixel_range = 10
    # for k in range(srcy_set.shape[0]):
    k = 0
    srcx = srcx_set[k]
    srcy = srcy_set[k]
    dstx = dstx_set[k]
    dsty = dsty_set[k]

    dragged_srcx = dstx + (ratio + 1) * (srcx - dstx)
    dragged_srcy = dsty + (ratio + 1) * (srcy - dsty)
    for dimy in range(height):
        for dimx in range(width):


            alpha = ((dimx - dragged_srcx) * (dstx - dragged_srcx) + (dimy - dragged_srcy) * (dsty - dragged_srcy)) / ((dragged_srcx - dstx)*(dragged_srcx - dstx) + (dragged_srcy - dsty)*(dragged_srcy - dsty) + 1e-5)

            d2src = np.sqrt((dragged_srcx - dimx)**2 + (dragged_srcx - dimy)**2)
            d2dst = np.sqrt((dstx - dimx)**2 + (dsty - dimy)**2)
            d2line = np.abs((dstx - srcx) * (srcy - dimy) - (srcx - dimx) * (dsty - srcy)) / (np.sqrt( (srcx - dstx)*(srcx - dstx) + (srcy - dsty)*(srcy - dsty) ) + 1e-5)

            if alpha < 1 and alpha > 0:
                dgeneral = d2line
            else:
                if d2src > d2dst:
                    dgeneral = d2dst
                else:
                    dgeneral = d2src


            alpha_weight = window_function(alpha, smooth_range)
            pixel_range_weight = distance_function(dgeneral, pixel_range)

            # recorder[k][0] = np.power(1.0 / (0.1 + dgeneral), 1)
            # recorder[k][1] = alpha_weight * alpha * (srcx - dstx) + dimx
            # recorder[k][2] = alpha_weight * alpha * (srcy - dsty) + dimy
            morphed_x[dimy][dimx] = pixel_range_weight * alpha_weight * alpha * (srcx - dstx)
            morphed_y[dimy][dimx] = pixel_range_weight * alpha_weight * alpha * (srcy - dsty)
            # totweights = 0
            # avex = 0
            # avey = 0
            # for k in range(srcy_set.shape[0]):
            #     totweights = totweights + recorder[k][0]
            # for k in range(srcy_set.shape[0]):
            #     avex = avex + recorder[k][0] / totweights * recorder[k][1]
            #     avey = avey + recorder[k][0] / totweights * recorder[k][2]
            # morphed_x[dimy][dimx] = avex
            # morphed_y[dimy][dimx] = avey
    morphed_sum = np.abs(morphed_x) + np.abs(morphed_y)
    vmax = 0.978
    morphed_sum = morphed_sum / vmax
    cm = plt.get_cmap('magma')
    morphed_sum = (cm(morphed_sum) * 255).astype(np.uint8)
    morphed_sum = pil.fromarray(morphed_sum)
    plt.figure()
    plt.imshow(morphed_sum)
    plt.plot([srcx, dstx], [srcy, dsty])
    plt.plot([srcx, dragged_srcx], [srcy, dragged_srcy])
    plt.scatter([dstx], [dsty], c = 'r')
    plt.scatter([dragged_srcx], [dragged_srcy], c = 'g')
    plt.scatter([srcx], [srcy], c = 'b')

    """
    return morphed_x, morphed_y


class BNMorphFunction(Function):
    @staticmethod
    def forward(ctx):
        return

    @staticmethod
    def backward(ctx):
        return

    @staticmethod
    def find_corresponding_pts(binMapsrc, binMapdst, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding):
        binMapsrc = binMapsrc.float()
        binMapdst = binMapdst.float()
        pixel_distance_weight = float(pixel_distance_weight)
        alpha_distance_weight = float(alpha_distance_weight)
        alpha_padding = float(alpha_padding)
        pixel_mulline_distance_weight = float(pixel_mulline_distance_weight)
        orgpts_x, orgpts_y, correspts_x, correspts_y, morphedx, morphedy = bnmorph_getcorpts.find_corespond_pts(binMapsrc, binMapdst, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding)
        ocoeff = dict()
        ocoeff['orgpts_x'] = orgpts_x
        ocoeff['orgpts_y'] = orgpts_y
        ocoeff['correspts_x'] = correspts_x
        ocoeff['correspts_y'] = correspts_y
        return morphedx, morphedy, ocoeff


    @staticmethod
    def find_corresponding_pts_debug(binMapsrc, binMapdst, disparityMap, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding, semantic_figure):
        binMapsrc = binMapsrc.float()
        binMapdst = binMapdst.float()
        pixel_distance_weight = float(pixel_distance_weight)
        alpha_distance_weight = float(alpha_distance_weight)
        alpha_padding = float(alpha_padding)
        pixel_mulline_distance_weight = float(pixel_mulline_distance_weight)
        orgpts_x, orgpts_y, correspts_x, correspts_y, morphedx, morphedy = bnmorph_getcorpts.find_corespond_pts(binMapsrc, binMapdst, xx, yy, sxx, syy, cxx, cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding)

        # height = disparityMap.shape[2]
        # width = disparityMap.shape[3]
        # morphedx = (morphedx / (width - 1) - 0.5) * 2
        # morphedy = (morphedy / (height - 1) - 0.5) * 2
        # grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
        # disparityMap_morphed = torch.nn.functional.grid_sample(disparityMap, grid, padding_mode="border")
        # tensor2disp(disparityMap_morphed, vmax=0.08, ind=0).show()

        # morphed_x = np.zeros([height, width])
        # morphed_y = np.zeros([height, width])
        # selector = orgpts_x[0, 0, :, :] > -1e-3
        # srcx_set = orgpts_x[0, 0, :, :][selector].cpu().numpy()
        # srcy_set = orgpts_y[0, 0, :, :][selector].cpu().numpy()
        # dstx_set = correspts_x[0, 0, :, :][selector].cpu().numpy()
        # dsty_set = correspts_y[0, 0, :, :][selector].cpu().numpy()
        # recorder = np.zeros([srcx_set.shape[0], 3])
        #
        # selector = srcx_set < -1e10
        # for i in range(srcx_set.shape[0]):
        #     if np.sqrt((srcx_set[i] - 238)**2 + (srcy_set[i] - 141)**2) < 1e10:
        #         selector[i] = 1
        # srcy_set = srcy_set[selector]
        # dstx_set = dstx_set[selector]
        # dsty_set = dsty_set[selector]
        # srcx_set = srcx_set[selector]
        # morphed_x, morphed_y = morph_along_lines(height, width, morphed_x, morphed_y, recorder, srcx_set, srcy_set, dstx_set, dsty_set)
        #
        # morphedx = torch.from_numpy(morphed_x).unsqueeze(0).unsqueeze(0).cuda().float()
        # morphedy = torch.from_numpy(morphed_y).unsqueeze(0).unsqueeze(0).cuda().float()
        # height = disparityMap.shape[2]
        # width = disparityMap.shape[3]
        # morphedx = (morphedx / (width - 1) - 0.5) * 2
        # morphedy = (morphedy / (height - 1) - 0.5) * 2
        # grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
        # disparityMap_morphed = torch.nn.functional.grid_sample(disparityMap, grid, padding_mode="border")
        # if semantic_figure is not None:
        #     fig_morphed = tensor2disp(disparityMap_morphed, vmax=0.08, ind=0)
        #     fig_disp = tensor2disp(disparityMap, vmax=0.08, ind=0)
        #     fig_morphed_overlayed = pil.fromarray((np.array(semantic_figure) * 0.5 + np.array(fig_morphed) * 0.5).astype(np.uint8))
        #     fig_disp_overlayed =  pil.fromarray((np.array(semantic_figure) * 0.5 + np.array(fig_disp) * 0.5).astype(np.uint8))
        #     fig_combined = pil.fromarray(np.concatenate([np.array(fig_disp_overlayed), np.array(fig_morphed_overlayed), np.array(fig_disp), np.array(fig_morphed)], axis=0))
        # else:
        #     fig_combined = None
        # return fig_combined, disparityMap_morphed
        return morphedx, morphedy


class BNMorph(nn.Module):
    def __init__(self, height, width, serachWidth = 7, searchHeight = 3, sparsityRad = 2, senseRange = 20, pixel_distance_weight = 20, alpha_distance_weight = 0.7, pixel_mulline_distance_weight = 15, alpha_padding = 0.6):
        super(BNMorph, self).__init__()
        self.height = height
        self.width = width
        self.searchWidth = serachWidth
        self.searchHeight = searchHeight
        self.sparsityRad = sparsityRad
        self.senseRange = senseRange
        self.pixel_distance_weight = pixel_distance_weight
        self.alpha_distance_weight = alpha_distance_weight
        self.alpha_padding = alpha_padding
        self.pixel_mulline_distance_weight = pixel_mulline_distance_weight

        self.pixel_distance_weight_store = None
        self.alpha_distance_weight_store = None
        self.pixel_mulline_distance_weight_store = None
        self.alpha_padding_store = None

        colsearchSpan = np.arange(-self.searchHeight, self.searchHeight + 1)
        rowsearchSpan = np.arange(-self.searchWidth, self.searchWidth + 1)
        xx, yy = np.meshgrid(rowsearchSpan, colsearchSpan)
        xx = xx.flatten()
        yy = yy.flatten()
        dist = xx**2 + yy**2
        sortedInd = np.argsort(dist)
        self.xx = torch.nn.Parameter(torch.from_numpy(xx[sortedInd]).float(), requires_grad=False)
        self.yy = torch.nn.Parameter(torch.from_numpy(yy[sortedInd]).float(), requires_grad=False)

        sparsittSpan = np.arange(-self.sparsityRad, self.sparsityRad + 1)
        sxx, syy = np.meshgrid(sparsittSpan, sparsittSpan)
        self.sxx = torch.nn.Parameter(torch.from_numpy(sxx.flatten()).float(), requires_grad=False)
        self.syy = torch.nn.Parameter(torch.from_numpy(syy.flatten()).float(), requires_grad=False)


        senseSpan = np.arange(-self.senseRange, self.senseRange + 1)
        cxx, cyy = np.meshgrid(senseSpan, senseSpan)
        cxx = cxx.flatten()
        cyy = cyy.flatten()
        dist = cxx ** 2 + cyy ** 2
        sortedInd = np.argsort(dist)
        self.cxx = torch.nn.Parameter(torch.from_numpy(cxx[sortedInd]).float(), requires_grad=False)
        self.cyy = torch.nn.Parameter(torch.from_numpy(cyy[sortedInd]).float(), requires_grad=False)

    def find_corresponding_pts_debug(self, binMapsrc, binMapdst, disparityMap, semantic_figure):
        return BNMorphFunction.find_corresponding_pts_debug(binMapsrc, binMapdst, disparityMap, self.xx, self.yy, self.sxx, self.syy, self.cxx, self.cyy, self.pixel_distance_weight, self.alpha_distance_weight, self.pixel_mulline_distance_weight, self.alpha_padding, semantic_figure)

    def find_corresponding_pts(self, binMapsrc, binMapdst, pixel_distance_weight = None, alpha_distance_weight = None, pixel_mulline_distance_weight = None, alpha_padding = None):
        if pixel_distance_weight is None:
            pixel_distance_weight = self.pixel_distance_weight

        if alpha_distance_weight is None:
            alpha_distance_weight = self.alpha_distance_weight

        if pixel_mulline_distance_weight is None:
            pixel_mulline_distance_weight = self.pixel_mulline_distance_weight

        if alpha_padding is None:
            alpha_padding = self.alpha_padding

        self.pixel_distance_weight_store = pixel_distance_weight
        self.alpha_distance_weight_store = alpha_distance_weight
        self.pixel_mulline_distance_weight_store = pixel_mulline_distance_weight
        self.alpha_padding_store = alpha_padding

        return BNMorphFunction.find_corresponding_pts(binMapsrc, binMapdst, self.xx, self.yy, self.sxx, self.syy, self.cxx, self.cyy, pixel_distance_weight, alpha_distance_weight, pixel_mulline_distance_weight, alpha_padding)
    def print_params(self):
        if self.pixel_distance_weight_store is None:
            self.pixel_distance_weight_store = self.pixel_distance_weight
        if self.alpha_distance_weight_store is None:
            self.alpha_distance_weight_store = self.alpha_distance_weight
        if self.pixel_mulline_distance_weight_store is None:
            self.pixel_mulline_distance_weight_store = self.pixel_mulline_distance_weight
        if self.alpha_padding_store is None:
            self.alpha_padding_store = self.alpha_padding
        print("Sparsity %f, pixel_distance_weight % f, alpha_distance_weight % f, pixel_mulline_distance_weight % f, alpha_padding % f" % (self.sparsityRad, self.pixel_distance_weight_store,  self.alpha_distance_weight_store, self.pixel_mulline_distance_weight_store, self.alpha_padding_store))

