# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from copy import deepcopy
from utils import *
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from cityscapesscripts.helpers.labels import trainId2label, id2label
from PIL import Image

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot

class SwitchBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(SwitchBlock, self).__init__()

        self.conv_pos = Conv3x3(in_channels, out_channels)
        self.conv_neg = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)
        # self.isInstanceNorm = isInstanceNorm
        # if self.isInstanceNorm:
        #     self.insNorm = nn.InstanceNorm2d(num_features = out_channels, affine=isAffine)

    def forward(self, x, switch_on = False):
        x_pos = self.conv_pos(x)
        x_neg = self.conv_neg(x)
        # if self.isInstanceNorm:
        #     x_pos = self.insNorm(x_pos)
        #     x_neg = self.insNorm(x_neg)
        pos = self.nonlin(x_pos)
        neg = self.nonlin(x_neg)
        if switch_on:
            out = pos - neg
        else:
            out = pos + neg
        out = self.nonlin(out)
        # out = self.conv(x)
        return out
class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()


        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords))

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width))

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1))

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords
    def forward_no_normalize(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        return pix_coords
def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    repeat_time = disp.shape[1]
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True).repeat(1,repeat_time,1,1)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True).repeat(1,repeat_time,1,1)

    # grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    # grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-grad_img_x)
    grad_disp_y *= torch.exp(-grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class ArbSSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self, kehight = 3, kwidth = 3):
        super(ArbSSIM, self).__init__()
        self.kwidth = kwidth
        self.kehight = kehight
        self.mu_x_pool   = nn.AvgPool2d((kehight, kwidth), 1)
        self.mu_y_pool   = nn.AvgPool2d((kehight, kwidth), 1)
        self.sig_x_pool  = nn.AvgPool2d((kehight, kwidth), 1)
        self.sig_y_pool  = nn.AvgPool2d((kehight, kwidth), 1)
        self.sig_xy_pool = nn.AvgPool2d((kehight, kwidth), 1)

        self.refl = nn.ReflectionPad2d((int((self.kwidth - 1) / 2), int((self.kwidth - 1) / 2), int((self.kehight - 1) / 2), int((self.kehight - 1) / 2)))

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

class Merge_MultDisp(nn.Module):
    def __init__(self, scales, semanType = 19, batchSize = 6, isMulChannel = False):
        # Merge multiple channel disparity to single channel according to semantic
        super(Merge_MultDisp, self).__init__()
        self.scales = scales
        self.semanType = semanType
        self.batchSize = batchSize
        self.sfx = nn.Softmax(dim=1).cuda()
        self.isMulChannel = isMulChannel
        # self.weights_time = 0

    def forward(self, inputs, outputs, eval = False):
        height = inputs[('color_aug', 0, 0)].shape[2]
        width = inputs[('color_aug', 0, 0)].shape[3]
        outputFormat = [self.batchSize, self.semanType + 1, height, width]

        if ('seman', 0) in outputs:
            for scale in self.scales:
                se_gt_name = ('seman', scale)
                seg = F.interpolate(outputs[se_gt_name], size=[height, width], mode='bilinear', align_corners=False)
                outputs[se_gt_name] = seg

        if self.isMulChannel:
            for scale in self.scales:
                disp_pred_name = ('mul_disp', scale)
                # disp = F.interpolate(outputs[disp_pred_name], [height, width], mode="bilinear", align_corners=False)
                disp = outputs[disp_pred_name]
                disp = torch.cat([disp, torch.mean(disp, dim=1, keepdim=True)], dim=1)
                outputs[disp_pred_name] = disp

            if 'seman_gt' in inputs and not eval:
                indexRef = deepcopy(inputs['seman_gt'])
                outputs['gtMask'] = indexRef != 255
                indexRef[indexRef == 255] = self.semanType
                disp_weights = torch.zeros(outputFormat).permute(0, 2, 3, 1).contiguous().view(-1, outputFormat[1]).cuda()
                indexRef = indexRef.permute(0, 2, 3, 1).contiguous().view(-1, 1)
                disp_weights[torch.arange(disp_weights.shape[0]), indexRef[:, 0]] = 1
                disp_weights = disp_weights.view(outputFormat[0], outputFormat[2], outputFormat[3],
                                                 outputFormat[1]).permute(0, 3, 1, 2)
                for scale in self.scales:
                    disp_weights = F.interpolate(disp_weights, [int(height / (2 ** scale)), int(width / (2 ** scale))],
                                                 mode="nearest")
                    outputs[('disp_weights', scale)] = disp_weights
            elif ('seman', 0) in outputs:
                # indexRef = torch.argmax(self.sfx(outputs[('seman', 0)]), dim=1, keepdim=True)
                disp_weights = torch.cat([self.sfx(outputs[('seman', 0)]),torch.zeros(outputFormat[0], outputFormat[2], outputFormat[3]).unsqueeze(1).cuda()], dim=1)
                for scale in self.scales:
                    disp_weights = F.interpolate(disp_weights, [int(height / (2 ** scale)), int(width / (2 ** scale))],
                                                 mode="bilinear", align_corners=False)
                    outputs[('disp_weights', scale)] = disp_weights

            # outputs['disp_weights'] = disp_weights
            for scale in self.scales:
                ref_name = ('mul_disp', scale)
                outputs[('disp', scale)] = torch.sum(outputs[ref_name] * outputs[('disp_weights', scale)], dim=1, keepdim=True)
        else:
            for scale in self.scales:
                ref_name = ('mul_disp', scale)
                if ref_name in outputs:
                    outputs[('disp', scale)] = outputs[ref_name]

class Compute_SemanticLoss(nn.Module):
    def __init__(self, classtype = 19, min_scale = 3):
        super(Compute_SemanticLoss, self).__init__()
        self.scales = list(range(4))[0:min_scale+1]
        # self.cen = nn.CrossEntropyLoss(reduction = 'none')
        self.cen = nn.CrossEntropyLoss(ignore_index = 255)
        self.classtype = classtype # default is cityscape setting 19
    def reorder(self, input, clssDim):
        return input.permute(2,3,1,0).contiguous().view(-1, clssDim)
    def forward(self, inputs, outputs, use_sep_semant_train = False):
        # height = inputs['seman_gt'].shape[2]
        # width = inputs['seman_gt'].shape[3]
        if not use_sep_semant_train:
            label = inputs['seman_gt']
        else:
            label = inputs['seperate_seman_gt']
        # Just for check
        # s = inputs['seman_gt'][0, 0, :, :].cpu().numpy()
        # visualize_semantic(s).show()
        # img = pil.fromarray((inputs[("color_aug", 0, 0)].permute(0,2,3,1)[0,:,:,:].cpu().numpy() * 255).astype(np.uint8))
        # img.show()
        # visualize_semantic(pred[0,:,:]).show()
        loss_toshow = dict()
        loss = 0
        for scale in self.scales:
            entry = ('seman', scale)
            scaled = outputs[entry]
            # scaled = F.interpolate(outputs[entry], size = [height, width], mode = 'bilinear')
            # rearranged = self.reorder(scaled, self.classtype)
            # cenl = self.cen(rearranged[mask[:,0], :], label[mask])
            cenl = self.cen(scaled, label.squeeze(1))
            loss_toshow["loss_seman/{}".format(scale)] = cenl
            loss = loss + cenl
            # just for check
            # m1 = rearranged[mask[:,0], :]
            # m2 = label[mask]
            # m3 = m1.gather(1, m2.view(-1,1))
            # loss_self = -log()
        loss = loss / len(self.scales)
        return loss, loss_toshow


class ComputeSurfaceNormal(nn.Module):
    def __init__(self, height, width, batch_size):
        super(ComputeSurfaceNormal, self).__init__()
        self.height = height
        self.width = width
        self.batch_size = batch_size
        # self.surnormType = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic sign', 'terrain']
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xx = xx.flatten().astype(np.float32)
        yy = yy.flatten().astype(np.float32)
        self.pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        self.pix_coords = torch.from_numpy(self.pix_coords).permute(0,2,1)
        self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        self.pix_coords = self.pix_coords.cuda()
        self.ones = self.ones.cuda()
        self.init_gradconv()

    def init_gradconv(self):
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)
        self.convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)

        self.convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convy.weight = nn.Parameter(weightsy,requires_grad=False)


    def forward(self, depthMap, invcamK):
        depthMap = depthMap.view(self.batch_size, -1)
        cam_coords = self.pix_coords * torch.stack([depthMap, depthMap, depthMap], dim=1)
        cam_coords = torch.cat([cam_coords, self.ones], dim=1)
        veh_coords = torch.matmul(invcamK, cam_coords)
        veh_coords = veh_coords.view(self.batch_size, 4, self.height, self.width)
        veh_coords = veh_coords
        changex = torch.cat([self.convx(veh_coords[:, 0:1, :, :]), self.convx(veh_coords[:, 1:2, :, :]), self.convx(veh_coords[:, 2:3, :, :])], dim=1)
        changey = torch.cat([self.convy(veh_coords[:, 0:1, :, :]), self.convy(veh_coords[:, 1:2, :, :]), self.convy(veh_coords[:, 2:3, :, :])], dim=1)
        surfnorm = torch.cross(changex, changey, dim=1)
        surfnorm = F.normalize(surfnorm, dim = 1)
        return surfnorm

    def visualize(self, depthMap, invcamK, orgEstPts = None, gtEstPts = None, viewindex = 0):
        # First compute 3d points in vehicle coordinate system
        depthMap = depthMap.view(self.batch_size, -1)
        cam_coords = self.pix_coords * torch.stack([depthMap, depthMap, depthMap], dim=1)
        cam_coords = torch.cat([cam_coords, self.ones], dim=1)
        veh_coords = torch.matmul(invcamK, cam_coords)
        veh_coords = veh_coords.view(self.batch_size, 4, self.height, self.width)
        veh_coords = veh_coords
        changex = torch.cat([self.convx(veh_coords[:, 0:1, :, :]), self.convx(veh_coords[:, 1:2, :, :]), self.convx(veh_coords[:, 2:3, :, :])], dim=1)
        changey = torch.cat([self.convy(veh_coords[:, 0:1, :, :]), self.convy(veh_coords[:, 1:2, :, :]), self.convy(veh_coords[:, 2:3, :, :])], dim=1)
        surfnorm = torch.cross(changex, changey, dim=1)
        surfnorm = F.normalize(surfnorm, dim = 1)

        # check
        # ckInd = 22222
        # x = self.pix_coords[viewindex, 0, ckInd].long()
        # y = self.pix_coords[viewindex, 1, ckInd].long()
        # ptsck = veh_coords[viewindex, :, y, x]
        # projecteck = torch.inverse(invcamK)[viewindex, :, :].cpu().numpy() @ ptsck.cpu().numpy().T
        # x_ = projecteck[0] / projecteck[2]
        # y_ = projecteck[1] / projecteck[2] # (x, y) and (x_, y_) should be equal

        # colorize this figure
        surfacecolor = surfnorm / 2 + 0.5
        img = surfacecolor[viewindex, :, :, :].permute(1,2,0).detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        # pil.fromarray(img).show()

        # objreg = ObjRegularization()
        # varloss = varLoss(scale=1, windowsize=7, inchannel=surfnorm.shape[1])
        # var = varloss(surfnorm)
        # if orgEstPts is not None and gtEstPts is not None:
        #     testPts = veh_coords[viewindex, :, :].permute(1,0).cpu().numpy()
        #     fig = plt.figure()
        #     ax = Axes3D(fig)
        #     ax.view_init(elev=6., azim=170)
        #     ax.dist = 4
        #     ax.scatter(orgEstPts[0::100, 0], orgEstPts[0::100, 1], orgEstPts[0::100, 2], s=0.1, c='b')
        #     ax.scatter(testPts[0::10, 0], testPts[0::10, 1], testPts[0::10, 2], s=0.1, c='g')
        #     ax.scatter(gtEstPts[0::100, 0], gtEstPts[0::100, 1], gtEstPts[0::100, 2], s=0.1, c='r')
        #     ax.set_zlim(-10, 10)
        #     plt.ylim([-10, 10])
        #     plt.xlim([10, 16])
        #     set_axes_equal(ax)
        return pil.fromarray(img)

class varLoss(nn.Module):
    def __init__(self, windowsize = 3, inchannel = 3):
        super(varLoss, self).__init__()
        assert windowsize % 2 != 0, "pls input odd kernel size"
        # self.scale = scale
        self.windowsize = windowsize
        self.inchannel = inchannel
        self.initkernel()
    def initkernel(self):
        # kernel is for mean value calculation
        weights = torch.ones((self.inchannel, 1, self.windowsize, self.windowsize))
        weights = weights / (self.windowsize * self.windowsize)
        self.conv = nn.Conv2d(in_channels=self.inchannel, out_channels=self.inchannel, kernel_size=self.windowsize, padding=0, bias=False, groups=self.inchannel)
        self.conv.weight = nn.Parameter(weights, requires_grad=False)
        # self.conv.cuda()
    def forward(self, input):
        pad = int((self.windowsize - 1) / 2)
        scaled = input[:,:, pad : -pad, pad : -pad] - self.conv(input)
        loss = scaled * scaled
        # loss = torch.mean(scaled * scaled)
        # check
        # ckr = self.conv(input)
        # exptime = 100
        # for i in range(exptime):
        #     batchind = torch.randint(0, ckr.shape[0], [1]).long()[0]
        #     chanind = torch.randint(0, ckr.shape[1], [1]).long()[0]
        #     xind = torch.randint(pad, ckr.shape[2]-pad, [1]).long()[0]
        #     yind = torch.randint(pad, ckr.shape[3]-pad, [1]).long()[0]
        #     ra = torch.mean(input[batchind, chanind, xind -pad : xind + pad + 1, yind - pad : yind + pad + 1])
        #     assert torch.abs(ra - ckr[batchind, chanind, xind - pad, yind - pad]) < 1e-5, "wrong"
        return loss
    def visualize(self, input):
        pad = int((self.windowsize - 1) / 2)
        scaled = input[:,:, pad : -pad, pad : -pad] - self.conv(input)
        errMap = scaled * scaled
        # loss = torch.mean(scaled * scaled)
        return errMap

class SelfOccluMask(nn.Module):
    def __init__(self, maxDisp = 21):
        super(SelfOccluMask, self).__init__()
        self.maxDisp = maxDisp
        self.pad = self.maxDisp
        self.init_kernel()
        self.boostfac = 400
    def init_kernel(self):
        # maxDisp is the largest disparity considered
        # added with being compated pixels
        convweights = torch.zeros(self.maxDisp, 1, 3, self.maxDisp + 2)
        for i in range(0, self.maxDisp):
            convweights[i, 0, :, 0:2] = 1/6
            convweights[i, 0, :, i+2:i+3] = -1/3
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=self.maxDisp, kernel_size=(3,self.maxDisp + 2), stride=1, padding=self.pad, bias=False)
        self.conv.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor) + 1, requires_grad=False)
        self.conv.weight = nn.Parameter(convweights, requires_grad=False)

        # convweights_opp = torch.flip(convweights, dims=[1])
        # self.conv_opp = torch.nn.Conv2d(in_channels=1, out_channels=self.maxDisp, kernel_size=(3,self.maxDisp + 2), stride=1, padding=self.pad, bias=False)
        # self.conv_opp.bias = nn.Parameter(torch.arange(self.maxDisp).type(torch.FloatTensor), requires_grad=False)
        # self.conv_opp.weight = nn.Parameter(convweights_opp, requires_grad=False)

        # self.weightck = (torch.sum(torch.abs(self.conv.weight)) + torch.sum(torch.abs(self.conv.bias)))
        # self.gausconv = get_gaussian_kernel(channels = 1, padding = 1)
        # self.gausconv.cuda()

        self.detectWidth = 19  # 3 by 7 size kernel
        # self.detectWidth = 41
        self.detectHeight = 3
        convWeightsLeft = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsRight = torch.zeros(1, 1, self.detectHeight, self.detectWidth)
        convWeightsLeft[0, 0, :, :int((self.detectWidth + 1) / 2)] = 1
        convWeightsRight[0, 0, :, int((self.detectWidth - 1) / 2):] = 1
        self.convLeft = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                        kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                        padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convRight = torch.nn.Conv2d(in_channels=1, out_channels=1,
                                         kernel_size=(self.detectHeight, self.detectWidth), stride=1,
                                         padding=[1, int((self.detectWidth - 1) / 2)], bias=False)
        self.convLeft.weight = nn.Parameter(convWeightsLeft, requires_grad=False)
        self.convRight.weight = nn.Parameter(convWeightsRight, requires_grad=False)
        self.th = 0.05
    def forward(self, dispmap, bsline):
        # dispmap = self.gausconv(dispmap)

        # assert torch.abs(self.weightck - (torch.sum(torch.abs(self.conv.weight)) + torch.sum(torch.abs(self.conv.bias)))) < 1e-2, "weights changed"
        with torch.no_grad():
            maskl = self.computeMask(dispmap, direction='l')
            maskr = self.computeMask(dispmap, direction='r')
            lind = bsline < 0
            rind = bsline > 0
            mask = torch.zeros_like(dispmap)
            mask[lind,:, :, :] = maskl[lind,:, :, :]
            mask[rind, :, :, :] = maskr[rind, :, :, :]

            return mask
    def computeMask(self, dispmap, direction):
        with torch.no_grad():
            width = dispmap.shape[3]
            if direction == 'l':
                # output = self.conv(dispmap)
                # output = torch.min(output, dim=1, keepdim=True)[0]
                # output = output[:,:,self.pad-1:-(self.pad-1):,-width:]
                # mask = torch.tanh(-output * self.boostfac)
                # mask = mask.masked_fill(mask < 0.9, 0)
                output = self.conv(dispmap)
                output = torch.clamp(output, max=0)
                output = torch.min(output, dim=1, keepdim=True)[0]
                output = output[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                output = torch.tanh(-output)
                mask = (output > self.th).float()
                # mask = (mask > 0.05).float()
            elif direction == 'r':
                # dispmap_opp = torch.flip(dispmap, dims=[3])
                # output_opp = self.conv(dispmap_opp)
                # output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
                # output_opp = output_opp[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                # mask = torch.tanh(-output_opp * self.boostfac)
                # mask = mask.masked_fill(mask < 0.9, 0)
                # mask = torch.flip(mask, dims=[3])
                dispmap_opp = torch.flip(dispmap, dims=[3])
                output_opp = self.conv(dispmap_opp)
                output_opp = torch.clamp(output_opp, max=0)
                output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
                output_opp = output_opp[:, :, self.pad - 1:-(self.pad - 1):, -width:]
                output_opp = torch.tanh(-output_opp)
                mask = (output_opp > self.th).float()
                mask = torch.flip(mask, dims=[3])

                # viewInd = 0
                # cm = plt.get_cmap('magma')
                # viewSSIMMask = mask[viewInd, 0, :, :].detach().cpu().numpy()
                # vmax = np.percentile(viewSSIMMask, 95)
                # viewSSIMMask = (cm(viewSSIMMask / vmax) * 255).astype(np.uint8)
                # pil.fromarray(viewSSIMMask).show()


                # viewdisp = dispmap[viewInd, 0, :, :].detach().cpu().numpy()
                # vmax = np.percentile(viewdisp, 90)
                # viewdisp = (cm(viewdisp / vmax) * 255).astype(np.uint8)
                # pil.fromarray(viewdisp).show()
            return mask
    def visualize(self, dispmap, viewind = 0):
        cm = plt.get_cmap('magma')

        width = dispmap.shape[3]
        output = self.conv(dispmap)
        output = torch.clamp(output, max=0)
        # output = torch.abs(output + 1)
        output = torch.min(output, dim=1, keepdim=True)[0]
        output = output[:,:,self.pad-1:-(self.pad-1):,-width:]
        output = torch.tanh(-output)
        mask = output
        mask = mask > 0.1
        # a = output[0,0,:,:].detach().cpu().numpy()
        # mask = torch.tanh(-output) + 1
        # mask = torch.tanh(-output * self.boostfac)
        # mask = mask.masked_fill(mask < 0.9, 0)

        dispmap_opp = torch.flip(dispmap, dims=[3])
        output_opp = self.conv(dispmap_opp)
        output_opp = torch.min(output_opp, dim=1, keepdim=True)[0]
        output_opp = output_opp[:,:,self.pad-1:-(self.pad-1):,-width:]
        # output = output[:,:,pad:-pad, pad:-pad]
        mask_opp = torch.tanh(-output_opp * self.boostfac)
        # mask_opp = torch.clamp(mask_opp, min=0)
        # mask_opp = mask_opp.masked_fill(mask_opp < 0.8, 0)
        mask_opp = mask_opp.masked_fill(mask_opp < 0.9, 0)
        mask_opp = torch.flip(mask_opp, dims=[3])

        # mask = (mask + mask_opp) / 2
        # mask[mask < 0] = 0

        binmask = mask > 0.1
        viewbin = binmask[viewind, 0, :, :].detach().cpu().numpy()
        # pil.fromarray((viewbin * 255).astype(np.uint8)).show()
        #
        # binmask_opp = mask_opp > 0.3
        # viewbin = binmask_opp[viewind, 0, :, :].detach().cpu().numpy()
        # pil.fromarray((viewbin * 255).astype(np.uint8)).show()

        viewmask = mask[viewind, 0, :, :].detach().cpu().numpy()
        viewmask = (cm(viewmask)* 255).astype(np.uint8)
        # pil.fromarray(viewmask).show()

        viewmask_opp = mask_opp[viewind, 0, :, :].detach().cpu().numpy()
        viewmask_opp = (cm(viewmask_opp)* 255).astype(np.uint8)
        # pil.fromarray(viewmask_opp).show()

        # dispmap = dispmap * (1 - mask)
        viewdisp = dispmap[viewind, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(viewdisp, 90)
        viewdisp = (cm(viewdisp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp).show()

        dispmap_sup = dispmap * (1 - mask.float())
        view_dispmap_sup = dispmap_sup[viewind, 0, :, :].detach().cpu().numpy()
        vmax = np.percentile(view_dispmap_sup, 90)
        view_dispmap_sup = (cm(view_dispmap_sup / vmax) * 255).astype(np.uint8)
        # pil.fromarray(view_dispmap_sup).show()

        # viewdisp_opp = dispmap_opp[viewind, 0, :, :].detach().cpu().numpy()
        # vmax = np.percentile(viewdisp_opp, 90)
        # viewdisp_opp = (cm(viewdisp_opp / vmax)* 255).astype(np.uint8)
        # pil.fromarray(viewdisp_opp).show()
        return pil.fromarray(viewmask), pil.fromarray(viewdisp)

    def betterLeftRigthOccluMask(self, occluMask, foregroundMask, direction):
        with torch.no_grad():
            if direction == 'l':
                mask = occluMask.clone()
                coSelected = (self.convLeft(occluMask) > 0) * (self.convRight(foregroundMask) > 0)
                mask[coSelected] = 1
                mask = mask * (1 - foregroundMask)
            elif direction == 'r':
                mask = occluMask.clone()
                coSelected = (self.convRight(occluMask) > 0) * (self.convLeft(foregroundMask) > 0)
                mask[coSelected] = 1
                mask = mask * (1 - foregroundMask)
            return mask
    def betterSelfOccluMask(self, occluMask, foregroundMask, bsline, dispPred = None):
        with torch.no_grad():
            maskl = self.betterLeftRigthOccluMask(occluMask, foregroundMask, direction='l')
            maskr = self.betterLeftRigthOccluMask(occluMask, foregroundMask, direction='r')
            lind = bsline < 0
            rind = bsline > 0
            mask = torch.zeros_like(occluMask)
            mask[lind,:, :, :] = maskl[lind,:, :, :]
            mask[rind, :, :, :] = maskr[rind, :, :, :]
            return mask

class MutuallyRegularizedBorders(nn.Module):
    def __init__(self, width, height, batchsize):
        super(MutuallyRegularizedBorders, self).__init__()
        self.width = width
        self.height = height
        self.batchsize = batchsize

        self.sampleTimes = 2000
        self.sampleDense = 500
        self.sampleWindow = 10
        self.itTime = 500
        self.eps = 1e-5
        self.minNum = 2

        self.channelInd = list()
        for i in range(self.batchsize):
            self.channelInd.append(torch.ones(self.sampleTimes) * i)
        self.channelInd = torch.cat(self.channelInd, dim=0).long().cuda()
    def forward(self, depthmap, foregroundMask, backgroundMask):
        a = 1
    def kmeansCluster(self, depthVals, foreCounts, backCounts):
        # validRe = foreCounts + backCounts > 0
        # foreCounts_re = foreCounts.clone().float()
        # backCounts_re = backCounts.clone().float()
        # foreCent = torch.sum(depthVals * foreCounts_re, dim=1, keepdim=True) / (torch.sum(foreCounts_re, 1, keepdim=True) + self.eps)
        # backCent = torch.sum(depthVals * backCounts_re, dim=1, keepdim=True) / (torch.sum(backCounts_re, 1, keepdim=True) + self.eps)
        #
        # for i in range(self.itTime):
        #     tmp_re = torch.abs(depthVals - foreCent.expand(-1, self.sampleDense)) >= torch.abs(depthVals - backCent.expand(-1, self.sampleDense))
        #     foreCounts_re = (validRe * tmp_re).float()
        #     backCounts_re = (validRe * (1 - tmp_re)).float()
        #     foreCent = torch.sum(depthVals * foreCounts_re, dim=1, keepdim=True) / (torch.sum(foreCounts_re, 1, keepdim=True) + self.eps)
        #     backCent = torch.sum(depthVals * backCounts_re, dim=1, keepdim=True) / (torch.sum(backCounts_re, 1, keepdim=True) + self.eps)
        #
        #     index = 5
        #     fckval = depthVals[index, :][foreCounts[index, :] == 1]
        #     bckval = depthVals[index, :][backCounts[index, :] == 1]
        # return foreCounts_re.byte(), backCounts_re.byte()


        # validRe = (foreCounts + backCounts) > 0
        # depthVals_ = depthVals * validRe.float()
        #
        # foreCounts_re = foreCounts.clone().float()
        # backCounts_re = backCounts.clone().float()
        # foreDepthVals = depthVals * foreCounts_re
        # backDepthVals = depthVals * backCounts_re + (1 - backCounts_re) * 1e10
        #
        # for i in range(self.itTime):
        #     tmp_border = (torch.max(foreDepthVals, dim=1, keepdim=True)[0] + torch.min(backDepthVals, dim=1, keepdim=True)[0])/2
        #     tmp_re = depthVals < tmp_border.expand(-1, self.sampleDense)
        #     foreCounts_re = (validRe * tmp_re).float()
        #     backCounts_re = (validRe * (1 - tmp_re)).float()
        #     foreDepthVals = depthVals * foreCounts_re
        #     backDepthVals = depthVals * backCounts_re + (1 - backCounts_re) * 1e10
        #
        #     index = 5
        #     fckval = depthVals[index, :][foreCounts[index, :] == 1]
        #     bckval = depthVals[index, :][backCounts[index, :] == 1]
        #
        #     fckval_sorted = depthVals[index, :][foreCounts_re[index, :] == 1]
        #     bckval_sorted = depthVals[index, :][backCounts_re[index, :] == 1]

        validRe = (foreCounts + backCounts) > 0
        valid_depthVals = depthVals * validRe.float()
        sampleY = torch.arange(0, validRe.shape[0])
        metrics = list()
        decisionBorders = list()
        for i in range(self.itTime):
            sampledIndexY = sampleY
            sampledIndexX = torch.LongTensor([i]).repeat(sampleY.shape)
            decisionBorder = valid_depthVals[sampledIndexY, sampledIndexX]
            decisionRe = valid_depthVals <= decisionBorder.unsqueeze(1).expand(-1,self.sampleDense)
            metric = torch.sum(decisionRe * (1 - foreCounts) * validRe, dim=1) +  torch.sum((1 - decisionRe) * foreCounts * validRe, dim=1)
            metrics.append(metric)
            decisionBorders.append(decisionBorder)
        metrics = torch.stack(metrics, dim=1)
        decisionBorders = torch.stack(decisionBorders, dim=1)
        minVal, minInd = torch.min(metrics, dim=1)
        finalBorder = decisionBorders[sampleY, minInd]

        tmp_re = depthVals <= finalBorder.unsqueeze(1).expand(-1, self.sampleDense)
        foreCounts_re = (validRe * tmp_re)
        backCounts_re = (validRe * (1 - tmp_re))

        # val_selector = ((torch.sum(foreCounts, 1)>self.minNum) * (torch.sum(backCounts, 1)>self.minNum)) > 0
        # iouFore = torch.sum(foreCounts_re * foreCounts, 1).float() / (
        #             torch.sum((foreCounts_re + foreCounts) > 0, 1).float() + self.eps)
        # iouBack = torch.sum(backCounts_re * backCounts, 1).float() / (torch.sum((backCounts_re + backCounts) > 0, 1).float() + self.eps)
        # print(iouFore[val_selector])
        # print(iouBack[val_selector])
        # viewind = 3
        # print(finalBorder[torch.sum(validRe, 1) > 2])
        # print(minVal[torch.sum(validRe, 1) > 2])
        # print(depthVals[viewind, :][foreCounts[viewind, :]])
        # print(depthVals[viewind, :][backCounts[viewind, :]])
        # print(depthVals[viewind, :][foreCounts_re[viewind, :]])
        # print(depthVals[viewind, :][backCounts_re[viewind, :]])
        # minVal[torch.sum(validRe, 3) > 2]
        return foreCounts_re, backCounts_re
    def visualization(self, depthmap, foregroundMask, backgroundMask, viewind = 0, rgb = None):
        sampledX = torch.LongTensor(self.batchsize * self.sampleTimes).random_(self.sampleWindow, self.width - self.sampleWindow)
        sampledY = torch.LongTensor(self.batchsize * self.sampleTimes).random_(self.sampleWindow, self.height - self.sampleWindow)
        valForeGroundPts = foregroundMask[self.channelInd, 0, sampledY, sampledX] == 1

        valx = sampledX[valForeGroundPts]
        valy = sampledY[valForeGroundPts]
        valChannel = self.channelInd[valForeGroundPts]

        valsampledX = valx.unsqueeze(1).expand(-1, self.sampleDense)
        valsampledY = valy.unsqueeze(1).expand(-1, self.sampleDense)
        valsampledChannel = valChannel.unsqueeze(1).expand(-1, self.sampleDense)
        valsampledX = valsampledX + torch.LongTensor(valsampledX.shape).random_(-self.sampleWindow, self.sampleWindow)
        valsampledY = valsampledY + torch.LongTensor(valsampledY.shape).random_(-self.sampleWindow, self.sampleWindow)

        valsampledChannel_flat = valsampledChannel.contiguous().view(-1)
        valsampledY_flat = valsampledY.contiguous().view(-1)
        valsampledX_flat = valsampledX.contiguous().view(-1)
        depthVals = depthmap[valsampledChannel_flat, 0, valsampledY_flat, valsampledX_flat]
        foreVals = foregroundMask[valsampledChannel_flat, 0, valsampledY_flat, valsampledX_flat]
        backVals = backgroundMask[valsampledChannel_flat, 0, valsampledY_flat, valsampledX_flat]
        foreCounts = (depthVals > 0) * (foreVals == 1)
        backCounts = (depthVals > 0) * (backVals == 1)
        foreCounts = foreCounts.view(-1, self.sampleDense)
        backCounts = backCounts.view(-1, self.sampleDense)
        depthVals = depthVals.view(-1, self.sampleDense)
        foreCounts_re, backCounts_re = self.kmeansCluster(depthVals, foreCounts, backCounts)
        iouFore = torch.sum(foreCounts_re * foreCounts, 1).float() / (torch.sum((foreCounts_re + foreCounts) > 0, 1).float() + self.eps)
        iouBack = torch.sum(backCounts_re * backCounts, 1).float() / (torch.sum((backCounts_re + backCounts) > 0, 1).float() + self.eps)

        # foreDepthMax = torch.max(depthVals * foreCounts.float(), dim=1)[0]
        # backDepthMin = torch.min(depthVals + 1e3 * (1 - backCounts.float()), dim=1)[0]
        # contrast = (foreDepthMax - backDepthMin) < 0
        foreDepthMean = torch.sum(depthVals * foreCounts.float(), dim=1) / (torch.sum(foreCounts.float(), dim=1) + self.eps)
        backDepthMean = torch.sum(depthVals * backCounts.float(), dim=1) / (torch.sum(backCounts.float(), dim=1) + self.eps)
        # contrast = (backDepthMean - foreDepthMean) > 0.1

        validCount = ((torch.sum(foreCounts, 1)>self.minNum) * (torch.sum(backCounts, 1)>self.minNum)) > 0
        # validCount = validCount * contrast
        validCount = validCount.float()
        iouForeMean = torch.sum(iouFore * validCount) / (torch.sum(validCount)+ self.eps)
        iouBackMean = torch.sum(iouBack * validCount)/ (torch.sum(validCount)+ self.eps)
        if torch.sum(validCount) > 0:
            isvalid = 1
        else:
            isvalid = 0
        return iouForeMean, iouBackMean, isvalid


        # viewind = 0
        #
        # disp_fig = tensor2disp(1 - depthmap, vmax=0.09, ind=0)
        # tensor2disp(foregroundMask, vmax=1, ind=0).show()
        # toShowSelector = valChannel == viewind
        # toshowX = valx[toShowSelector].detach().cpu().numpy()
        # toshowY = valy[toShowSelector].detach().cpu().numpy()
        # rgb_fig = np.array(tensor2rgb(rgb, viewind))
        # plt.imshow(rgb_fig)
        # plt.scatter(toshowX, toshowY, s = 0.7)
        # plt.show()
        #
        # allPoss = torch.arange(0, validCount.shape[0])
        # allPoss = allPoss[(validCount == 1) * (valChannel == viewind)]
        # toView = allPoss[torch.LongTensor([1]).random_(0, allPoss.shape[0])]
        # drawSelFore = foreCounts[toView, :]
        # drawX_fore = valsampledX[toView, :][drawSelFore].detach().cpu().numpy()
        # drawY_fore = valsampledY[toView, :][drawSelFore].detach().cpu().numpy()
        #
        # drawSelBack = backCounts[toView, :]
        # drawX_back = valsampledX[toView, :][drawSelBack].detach().cpu().numpy()
        # drawY_back = valsampledY[toView, :][drawSelBack].detach().cpu().numpy()
        #
        #
        # plt.close()
        # plt.figure()
        # plt.imshow(rgb_fig)
        # plt.scatter(drawX_fore, drawY_fore, s = 0.7, c = 'r')
        # plt.scatter(drawX_back, drawY_back, s=0.7, c='b')
        # plt.show()
        #
        #
        # drawSelFore = foreCounts_re[toView, :]
        # drawX_fore = valsampledX[toView, :][drawSelFore].detach().cpu().numpy()
        # drawY_fore = valsampledY[toView, :][drawSelFore].detach().cpu().numpy()
        #
        # drawSelBack = backCounts_re[toView, :]
        # drawX_back = valsampledX[toView, :][drawSelBack].detach().cpu().numpy()
        # drawY_back = valsampledY[toView, :][drawSelBack].detach().cpu().numpy()
        # plt.figure()
        # plt.imshow(rgb_fig)
        # plt.scatter(drawX_fore, drawY_fore, s = 0.7, c = 'r')
        # plt.scatter(drawX_back, drawY_back, s=0.7, c='c')
        # plt.show()
        #
        # print(iouFore[toView])
        # print(iouBack[toView])
        # print(depthVals[toView, :][drawSelBack].detach().cpu().numpy())
        # print(depthVals[toView, :][drawSelFore].detach().cpu().numpy())
        # plt.figure()
        # drawSelBack = backCounts[toView, :]
        # drawSelFore = foreCounts[toView, :]
        # a = depthVals[toView, :][drawSelFore].detach().cpu().numpy()
        # b = depthVals[toView, :][drawSelBack].detach().cpu().numpy()
        # plt.scatter(a, np.zeros_like(a), s = 0.7, c = 'r')
        # plt.scatter(b, np.zeros_like(b), s=0.7, c='b')
        #
        # plt.figure()
        # drawSelBack = backCounts_re[toView, :]
        # drawSelFore = foreCounts_re[toView, :]
        # a = depthVals[toView, :][drawSelFore].detach().cpu().numpy()
        # b = depthVals[toView, :][drawSelBack].detach().cpu().numpy()
        # plt.scatter(a, np.zeros_like(a), s = 0.7, c = 'r')
        # plt.scatter(b, np.zeros_like(b), s=0.7, c='c')
        #
        # print(metrics[toView, :][depthVals[toView, :] > 0])
        # plt.figure()
        # plt.imshow(disp_fig)
        # plt.scatter(drawX_fore, drawY_fore, s = 0.7, c = 'r')
        # plt.scatter(drawX_back, drawY_back, s=0.7, c='c')
        # plt.show()
        #
        # return iouForeMean, iouBackMean


class computeGradient(nn.Module):
    def __init__(self):
        super(computeGradient, self).__init__()
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [ 1.,  2.,  1.],
                                [ 0.,  0.,  0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy,requires_grad=False)


        self.approxWidth = 7
        middleInd = int((self.approxWidth - 1) / 2)
        weightsxl = torch.zeros([1, 1, 3, self.approxWidth])
        for i in range(middleInd):
            weightsxl[0,0,:,i] = torch.Tensor([-1.,-2.,-1.])

        weightsxr = torch.zeros([1, 1, 3, self.approxWidth])
        for i in range(middleInd):
            weightsxr[0,0,:,-i] = torch.Tensor([1.,2.,1.])

        weightsyu = torch.zeros([1, 1, 3, self.approxWidth])
        for i in range(self.approxWidth):
            weightsyu[0,0,0,i] = 1

        weightsyd = torch.zeros([1, 1, 3, self.approxWidth])
        for i in range(self.approxWidth):
            weightsyd[0,0,-1,i] = -1

        self.convDispxl = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,self.approxWidth), padding=(1, middleInd), bias=False)
        self.convDispxr = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,self.approxWidth), padding=(1, middleInd), bias=False)
        self.convDispyu = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,self.approxWidth), padding=(1, middleInd), bias=False)
        self.convDispyd = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3,self.approxWidth), padding=(1, middleInd), bias=False)
        self.convDispxl.weight = nn.Parameter(weightsxl,requires_grad=False)
        self.convDispxr.weight = nn.Parameter(weightsxr,requires_grad=False)
        self.convDispyu.weight = nn.Parameter(weightsyu,requires_grad=False)
        self.convDispyd.weight = nn.Parameter(weightsyd,requires_grad=False)


        # weightsxl = torch.Tensor([
        #                         [-1., 0., 0.],
        #                         [-2., 0., 0.],
        #                         [-1., 0., 0.]]).unsqueeze(0).unsqueeze(0)
        # weightsxr = torch.Tensor([
        #                         [0., 0., 1.],
        #                         [0., 0., 2.],
        #                         [0., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        #
        # weightsyu = torch.Tensor([
        #                         [ 1.,  2.,  1.],
        #                         [ 0.,  0.,  0.],
        #                         [ 0.,  0.,  0.]]).unsqueeze(0).unsqueeze(0)
        # weightsyd = torch.Tensor([
        #                         [ 0.,  0.,  0.],
        #                         [ 0.,  0.,  0.],
        #                         [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        #
        # self.convDispxl = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        # self.convDispxr = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        # self.convDispyu = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        # self.convDispyd = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        # self.convDispxl.weight = nn.Parameter(weightsxl,requires_grad=False)
        # self.convDispxr.weight = nn.Parameter(weightsxr,requires_grad=False)
        # self.convDispyu.weight = nn.Parameter(weightsyu,requires_grad=False)
        # self.convDispyd.weight = nn.Parameter(weightsyd,requires_grad=False)
        self.eps = 1e-3
    def computegrad11(self, inputMap):
        output = torch.abs(self.convDispx(inputMap)) + torch.abs(self.convDispy(inputMap))
        output[:, :, 0, :] = 0
        output[:, :, -1, :] = 0
        output[:, :, :, 0] = 0
        output[:, :, :, -1] = 0
        # tensor2disp(output, ind=0, percentile=96).show()
        return output
    def computegrad11_sparse(self, inputMap):
        inputMap_bin = (inputMap > 1e-1).float()
        inputMapxl = self.convDispxl(inputMap) / (torch.abs(self.convDispxl(inputMap_bin)) + self.eps)
        inputMapxr = self.convDispxr(inputMap) / (torch.abs(self.convDispxr(inputMap_bin)) + self.eps)
        inputMapyu = self.convDispyu(inputMap) / (torch.abs(self.convDispyu(inputMap_bin)) + self.eps)
        inputMapyd = self.convDispyd(inputMap) / (torch.abs(self.convDispyd(inputMap_bin)) + self.eps)

        inputmapx = (inputMapxl + inputMapxr) * ((torch.abs(inputMapxl) > self.eps) * (torch.abs(inputMapxr) > self.eps)).float()
        inputmapy = (inputMapyu + inputMapyd) * ((torch.abs(inputMapyu) > self.eps) * (torch.abs(inputMapyd) > self.eps)).float()
        inputmapGrad = torch.abs(inputmapx) + torch.abs(inputmapy)
        # tensor2disp(inputmapGrad, vmax=15, ind=0).show()
        # tensor2disp(torch.abs(inputMap) > self.eps, vmax=1, ind=0).show()
        return inputmapGrad


class computeBorderDistance(nn.Module):
    def __init__(self):
        super(computeBorderDistance, self).__init__()
        self.searchRange = 13
        self.convSet = list()
        for i in range(0,self.searchRange):
            tmpWeight = torch.ones([1,1,i*2 + 1,i * 2 + 1])
            tmpconv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=i*2 + 1,
                                        padding=i, bias=False)
            tmpconv.weight = nn.Parameter(tmpWeight,requires_grad=False)
            self.convSet.append(tmpconv.cuda())

    def computeDistance(self, source, dst):
        reset = list()
        source = source.float()
        dst = dst.float()
        for i in range(len(self.convSet)):
            tmpre = self.convSet[i](dst)
            tmpre = (tmpre > 0).float() * (self.searchRange + 1 - i)
            reset.append(tmpre)
        reset.append(torch.ones_like(tmpre))
        reset = torch.cat(reset, dim=1)
        reset = torch.max(reset, dim=1, keepdim=True)[0]
        # reset = reset * source

        a = torch.histc(reset[source == 1], min=0.99, max=self.searchRange + 2, bins=self.searchRange + 1)
        b = a.cpu().numpy()
        b = np.flip(b)
        # b = b / np.sum(b)
        # fig, ax = plt.subplots()
        # ax.bar(np.arange(len(b)), b, label='obj:car')
        # ax.set_ylabel('Percentile')
        # ax.set_xlabel('Distance in pixel')

        # tensor2disp(source, vmax=1, ind=0).show()
        # tensor2disp(dst, vmax=1, ind=0).show()
        # tensor2disp(reset, vmax=31, ind=0).show()
        # tensor2disp(reset > 30, vmax=1, ind=0).show()
        return b
class SecondOrderGrad(nn.Module):
    def __init__(self):
        super(SecondOrderGrad, self).__init__()
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [ 1.,  2.,  1.],
                                [ 0.,  0.,  0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy,requires_grad=False)
    def computegrad11(self, inputMap):
        output = torch.abs(self.convDispx(self.convDispx(inputMap))) + torch.abs(self.convDispy(self.convDispy(inputMap)))
        output[:, :, 0:2, :] = 0
        output[:, :, -1:-3, :] = 0
        output[:, :, :, 0:2] = 0
        output[:, :, :, -1:-3] = 0
        # tensor2disp(output, ind=0, percentile=96).show()
        return output


class BorderConverge(nn.Module):
    def __init__(self, height, width, batchSize):
        super(BorderConverge, self).__init__()
        weightsx = torch.Tensor([
                                [-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
                                [ 1.,  2.,  1.],
                                [ 0.,  0.,  0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy,requires_grad=False)

        self.depthTh = 0.011
        self.semanTh = 0.6

        self.height = height
        self.width = width
        self.batchSize = batchSize

        self.searchRange = 11
        weightsSearch = torch.ones([1, 1, 2 * self.searchRange + 1, 2 * self.searchRange + 1])
        self.searchKernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * self.searchRange + 1, padding=self.searchRange, bias=False)
        self.searchKernel.weight = nn.Parameter(weightsSearch, requires_grad=False)

        self.rangeMask = torch.zeros([1,1,self.height, self.width], dtype=torch.uint8, device=torch.device("cuda")).expand(self.batchSize, -1, -1, -1)
        self.rangeMask[:,:,self.searchRange:-self.searchRange,self.searchRange:-self.searchRange] = 1
        self.sampleDense = 2000
        self.channelInd = list()
        for i in range(self.batchSize):
            self.channelInd.append(torch.ones([self.sampleDense], dtype=torch.long, device=torch.device("cuda")) * (i))
        self.channelInd = torch.cat(self.channelInd, dim=0)

        aranged = torch.arange(-self.searchRange, self.searchRange + 1)
        addxx, addyy = torch.meshgrid([aranged, aranged])
        self.addxx = addxx.contiguous().view(-1).cuda()
        self.addyy = addyy.contiguous().view(-1).cuda()

        self.repNum = (self.searchRange*2 +1) * (self.searchRange*2+1)

    def forward(self, disparity, semantics):

        disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
        semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))

        disparity_grad_bin = disparity_grad > self.depthTh
        semantics_grad_bin = semantics_grad > self.semanTh

        with torch.no_grad():
            validPos = (self.searchKernel(semantics_grad_bin.float()) > 0) * disparity_grad_bin * self.rangeMask

            sampledx = torch.LongTensor(self.batchSize * self.sampleDense).random_(0, self.width).cuda()
            sampledy = torch.LongTensor(self.batchSize * self.sampleDense).random_(0, self.height).cuda()
            selector1 = validPos[self.channelInd, 0, sampledy, sampledx] == 1

            sampledx_selected = sampledx[selector1]
            sampledy_selected = sampledy[selector1]
            channelInd_selected = self.channelInd[selector1]

            remainedNum = sampledx_selected.shape[0]
            sampledx_pts = sampledx_selected.view(-1,1).expand(-1,self.repNum) + self.addxx.unsqueeze(0).expand(remainedNum, -1)
            sampledy_pts = sampledy_selected.view(-1,1).expand(-1,self.repNum) + self.addyy.unsqueeze(0).expand(remainedNum, -1)
            channelInd_pts = channelInd_selected.view(-1,1).expand(-1,self.repNum)
        if channelInd_pts is None:
            return 0
        vecs1 = disparity_grad[channelInd_pts, 0, sampledy_pts, sampledx_pts]
        vecs2 = semantics_grad[channelInd_pts, 0, sampledy_pts, sampledx_pts]
        vecs_bin = semantics_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]

        vecs1 = vecs1 * vecs_bin.float()
        vecs2 = vecs2 * vecs_bin.float()

        vecs1 = (vecs1 - torch.mean(vecs1, dim=1, keepdim=True).expand(-1, self.repNum))
        vecs1 = vecs1 / torch.norm(vecs1, dim=1, keepdim=True).expand(-1, self.repNum)
        vecs2 = (vecs2 - torch.mean(vecs2, dim=1, keepdim=True).expand(-1, self.repNum))
        vecs2 = vecs2 / torch.norm(vecs2, dim=1, keepdim=True).expand(-1, self.repNum)

        loss = torch.mean(torch.sum(vecs1 * vecs2, dim=1))

        return loss
    def visualization(self, disparity, semantics):
        disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
        semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))

        disparity_grad_bin = disparity_grad > self.depthTh
        semantics_grad_bin = semantics_grad > self.semanTh

        validPos = (self.searchKernel(semantics_grad_bin.float()) > 0) * disparity_grad_bin * self.rangeMask

        sampledx = torch.LongTensor(self.batchSize * self.sampleDense).random_(0, self.width).cuda()
        sampledy = torch.LongTensor(self.batchSize * self.sampleDense).random_(0, self.height).cuda()
        selector1 = validPos[self.channelInd, 0, sampledy, sampledx] == 1

        sampledx_selected = sampledx[selector1]
        sampledy_selected = sampledy[selector1]
        channelInd_selected = self.channelInd[selector1]

        remainedNum = sampledx_selected.shape[0]
        sampledx_pts = sampledx_selected.view(-1,1).expand(-1,self.repNum) + self.addxx.unsqueeze(0).expand(remainedNum, -1)
        sampledy_pts = sampledy_selected.view(-1,1).expand(-1,self.repNum) + self.addyy.unsqueeze(0).expand(remainedNum, -1)
        channelInd_pts = channelInd_selected.view(-1,1).expand(-1,self.repNum)

        vecs1 = disparity_grad[channelInd_pts, 0, sampledy_pts, sampledx_pts]
        # vecs1_bin = disparity_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]
        vecs2 = semantics_grad[channelInd_pts, 0, sampledy_pts, sampledx_pts]
        # vecs2_bin = semantics_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]
        vecs_bin = semantics_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]

        vecs1 = vecs1 * vecs_bin.float()
        vecs2 = vecs2 * vecs_bin.float()

        vecs1 = (vecs1 - torch.mean(vecs1, dim=1, keepdim=True).expand(-1, self.repNum))
        vecs1 = vecs1 / torch.norm(vecs1, dim=1, keepdim=True).expand(-1, self.repNum)
        vecs2 = (vecs2 - torch.mean(vecs2, dim=1, keepdim=True).expand(-1, self.repNum))
        vecs2 = vecs2 / torch.norm(vecs2, dim=1, keepdim=True).expand(-1, self.repNum)

        loss = torch.mean(torch.sum(vecs1 * vecs2, dim=1))

        tensor2disp(semantics_grad_bin.detach(), ind=0, vmax=1).show()
        tensor2disp(disparity_grad.detach(), ind=0, percentile=96).show()
        tensor2disp(semantics_grad.detach(), ind=0, percentile=96).show()
        tensor2disp(disparity_grad_bin.detach(), ind=0, vmax=1).show()
        tensor2disp(validPos.detach(), ind=0, vmax=1).show()

        viewind = 0
        drawSelector = channelInd_selected == viewind
        drawx = sampledx_selected[drawSelector].detach().cpu().numpy()
        drawy = sampledy_selected[drawSelector].detach().cpu().numpy()

        sampledx_pts_draw = sampledx_pts[drawSelector, :].detach().cpu().numpy()
        sampledy_pts_draw = sampledy_pts[drawSelector, :].detach().cpu().numpy()
        fig_disp = tensor2disp(disparity_grad.detach(), ind=0, percentile=96)
        plt.figure()
        plt.imshow(fig_disp)
        plt.scatter(drawx, drawy, c = 'g', s = 0.5)
        plt.scatter(sampledx_pts_draw.flatten(), sampledy_pts_draw.flatten(), c='b', s=0.5)
        plt.show()

        fig_disp_bin = tensor2disp(disparity_grad_bin.detach(), ind=0, percentile=96)
        plt.figure()
        plt.imshow(fig_disp_bin)
        plt.scatter(drawx, drawy, c = 'g', s = 0.5)
        plt.scatter(sampledx_pts_draw.flatten(), sampledy_pts_draw.flatten(), c='b', s=0.5)
        plt.show()

        toshowind = torch.arange(0, vecs_bin.shape[0])[drawSelector]
        toshowind = toshowind[torch.LongTensor(1).random_(0, toshowind.shape[0])][0]
        drawx_s = sampledx_pts[toshowind, :][vecs_bin[toshowind, :] == 1].detach().cpu().numpy()
        drawy_s = sampledy_pts[toshowind, :][vecs_bin[toshowind, :] == 1].detach().cpu().numpy()
        fig_seman_bin = tensor2disp(semantics_grad_bin.detach(), ind=0, percentile=96)
        plt.figure()
        plt.imshow(fig_seman_bin)
        plt.scatter(drawx_s, drawy_s, c = 'g', s = 0.5)
        plt.show()


class expBinaryMap(nn.Module):
    def __init__(self, height, width, batchSize):
        super(expBinaryMap, self).__init__()
        weightsx = torch.Tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy, requires_grad=False)

        self.depthTh = 0.011
        self.semanTh = 0.6

        self.height = height
        self.width = width
        self.batchSize = batchSize

        self.searchRange = 11
        weightsSearch = torch.ones([1, 1, 2 * self.searchRange + 1, 2 * self.searchRange + 1])
        self.searchKernel = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2 * self.searchRange + 1,
                                      padding=self.searchRange, bias=False)
        self.searchKernel.weight = nn.Parameter(weightsSearch, requires_grad=False)
        self.sampleDense = 20000

        self.channelInd = list()
        for i in range(self.batchSize):
            self.channelInd.append((torch.Tensor([i])).long().view([1,1,1,1]).expand(1, 1, self.height, self.width).cuda())
        self.channelInd = torch.cat(self.channelInd, dim=0)

        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        self.xx = torch.from_numpy(xx).view([1,1,self.height, self.width]).expand(self.batchSize, -1,-1,-1).long().cuda()
        self.yy = torch.from_numpy(yy).view([1,1,self.height, self.width]).expand(self.batchSize, -1,-1,-1).long().cuda()

        self.repNum = (self.searchRange * 2 + 1) * (self.searchRange * 2 + 1)
        self.eps = 1e-5
        self.sig = nn.Sigmoid()

        self.local_grad_search = 5
        self.maxpool = nn.MaxPool2d(2 * self.local_grad_search + 1, stride=1, padding=self.local_grad_search)

        aranged = torch.arange(-self.local_grad_search, self.local_grad_search + 1)
        addxx, addyy = torch.meshgrid([aranged, aranged])
        self.addxx = addxx.contiguous().view(-1).cuda()
        self.addyy = addyy.contiguous().view(-1).cuda()
        self.expandedNum = self.addxx.shape[0]

        self.rangeMask = torch.zeros([1, 1, self.height, self.width], dtype=torch.uint8,
                                     device=torch.device("cuda")).expand(self.batchSize, -1, -1, -1)
        self.rangeMask[:, :, self.local_grad_search:-self.local_grad_search, self.local_grad_search:-self.local_grad_search] = 1
        self.itNum = 3

        self.rec = {}
    def forward(self, disparity, semantics):
        disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
        semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))


        disparity_grad_sig = self.sig((disparity_grad - self.depthTh))
        semantics_grad_sig = self.sig((semantics_grad - self.semanTh))

        disparity_grad_bin = (disparity_grad_sig > 0.5).float()
        semantics_grad_bin = (semantics_grad_sig > 0.5).float()
        joint = ((disparity_grad_bin + semantics_grad_bin) > self.eps).float()
        # validPos = ((self.searchKernel(semantics_grad_bin) > self.eps).float() * (disparity_grad_bin > self.eps).float() * self.rangeMask.float()).detach()
        validPos = (
                (self.searchKernel(semantics_grad_bin) > self.eps).float() *
                (self.searchKernel(disparity_grad_bin) > 5).float() *
                joint *
                self.rangeMask.float()
                    ).detach().byte()

        semantics_valid_bin = semantics_grad_bin[validPos]
        disparity_grad_sig_valid = disparity_grad_sig[validPos]
        semantics_grad_sig_valid = semantics_grad_sig[validPos]
        upFlow = ((semantics_valid_bin) > 0).float()
        downFlow = 1 - upFlow
        loss = torch.sum(torch.abs(disparity_grad_sig_valid - semantics_grad_sig_valid) * upFlow) / torch.sum(upFlow + self.eps) + \
               torch.sum(torch.abs(disparity_grad_sig_valid - semantics_grad_sig_valid) * downFlow) / torch.sum(downFlow + self.eps)
        # loss = torch.sum(validPos * torch.abs(disparity_grad_sig_valid - semantics_grad_sig_valid)) / torch.sum(validPos + self.eps)

        # tensor2disp(disparity_grad_bin.detach() > 0.5, ind=0, vmax = 1).show()
        # tensor2disp(semantics_grad_bin.detach() > 0.5, ind=0, vmax=1).show()
        # tensor2disp(validPos.detach() > 0, ind=0, vmax = 1).show()
        # combinedChannels = torch.cat([semantics_grad_bin, validPos, torch.zeros_like(validPos)], dim=1)
        # tensor2rgb(combinedChannels.detach(), ind=0).show()
        # tensor2disp(torch.abs(disparity_grad_sig - semantics_grad_sig), ind=0, vmax=1).show()
        # tensor2disp(disparity_grad_sig, ind=0, vmax=0.6).show()
        return loss
    def visualizeForDebug(self, disparity, semantics):
        disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
        semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))

        disparity_grad_sig = self.sig((disparity_grad - self.depthTh))
        semantics_grad_sig = self.sig((semantics_grad - self.semanTh))

        disparity_grad_bin = (disparity_grad_sig > 0.5).float()
        semantics_grad_bin = (semantics_grad_sig > 0.5).float()

        # fig_disp = tensor2disp(disparity_grad_bin.detach() > 0.5, ind=0, vmax = 1)
        # fig_seman = tensor2disp(semantics_grad_bin.detach() > 0.5, ind=0, vmax=1)
        combinedChannels = torch.cat([disparity_grad_bin, semantics_grad_bin, torch.zeros_like(semantics_grad_bin)], dim=1)
        return tensor2rgb(combinedChannels.detach(), ind=0)
    def visualization(self, disparity, semantics):
        disparity_init = disparity.clone()
        semantics_init = semantics.clone()
        disparity = nn.Parameter(disparity_init.clone(), requires_grad=True)
        semantics = nn.Parameter(semantics_init.clone(), requires_grad=True)

        itNum = 5000
        model_optimizer = torch.optim.SGD([disparity, semantics], lr=1e3)
        for i in range(itNum):
            disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
            semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))


            disparity_grad_sig = self.sig((disparity_grad - self.depthTh))
            semantics_grad_sig = self.sig((semantics_grad - self.semanTh))

            disparity_grad_bin = (disparity_grad_sig > 0.5).float()
            semantics_grad_bin = (semantics_grad_sig > 0.5).float()

            # print(torch.mean(semantics_grad_bin))
            # print(torch.var(semantics_grad_bin))
            # print(torch.mean(disparity_grad_bin))
            # print(torch.var(disparity_grad_bin))
            # tensor2disp(disparity_grad_bin.detach() > 0.5, ind=0, vmax = 1).show()
            # tensor2disp(semantics_grad_bin.detach() > 0.5, ind=0, vmax=1).show()
            # print(torch.mean(disparity_grad_sig[disparity_grad_sig > 0.5]))
            # print(torch.mean(semantics_grad_sig[semantics_grad_bin > 0.5]))
            # print(torch.var(disparity_grad_sig[disparity_grad_sig > 0.5]))
            # print(torch.var(semantics_grad_sig[semantics_grad_bin > 0.5]))

            validPos = (self.searchKernel(semantics_grad_bin) > self.eps).float() * (
                        disparity_grad_bin > self.eps).float() * self.rangeMask.float()
            loss = torch.sum(validPos * torch.abs(disparity_grad_sig - semantics_grad_sig)) / torch.sum(validPos + self.eps)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            print("loss is %f" % loss)

            if np.mod(i, 10) == 0:
                fig_disp_grad = tensor2disp(disparity_grad_bin.detach() - 0.5, ind=0, vmax=0.5)
                fig_seman_grad = tensor2disp(semantics_grad_bin.detach() - 0.5, ind=0, vmax=0.5)
                fig_disp = tensor2disp(disparity.detach(), ind=0, vmax=0.11)
                fig_seman = tensor2disp(semantics, ind=0, vmax=1)
                pil.fromarray(np.concatenate([np.array(fig_disp), np.array(fig_seman), np.array(fig_disp_grad), np.array(fig_seman_grad)], axis=0)).save("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderMerge2/" + str(i) + ".png")

        a = 1
        torch.mean(torch.abs(disparity_init - disparity))
        torch.mean(torch.abs(semantics_init - semantics))
            # vecs_bin = semantics_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]
            #
            # vecs1 = vecs1 * vecs_bin.float()
            # vecs2 = vecs2 * vecs_bin.float()
            #
            # vecs1 = (vecs1 - torch.mean(vecs1, dim=1, keepdim=True).expand(-1, self.repNum))
            # vecs1 = vecs1 / torch.norm(vecs1, dim=1, keepdim=True).expand(-1, self.repNum)
            # vecs2 = (vecs2 - torch.mean(vecs2, dim=1, keepdim=True).expand(-1, self.repNum))
            # vecs2 = vecs2 / torch.norm(vecs2, dim=1, keepdim=True).expand(-1, self.repNum)

            # loss = torch.mean(torch.sum(vecs1 * vecs2, dim=1))

            # tensor2disp(semantics_grad_bin.detach(), ind=0, vmax=1).show()
            # tensor2disp(disparity_grad.detach(), ind=0, percentile=96).show()
            # tensor2disp(semantics_grad.detach(), ind=0, percentile=96).show()
            # tensor2disp(disparity_grad_bin.detach(), ind=0, vmax=1).show()
            # tensor2disp(validPos.detach(), ind=0, vmax=1).show()
            #
            # viewind = 0
            # drawSelector = channelInd_selected == viewind
            # drawx = sampledx_selected[drawSelector].detach().cpu().numpy()
            # drawy = sampledy_selected[drawSelector].detach().cpu().numpy()
            #
            # sampledx_pts_draw = sampledx_pts[drawSelector, :].detach().cpu().numpy()
            # sampledy_pts_draw = sampledy_pts[drawSelector, :].detach().cpu().numpy()
            # fig_disp = tensor2disp(disparity_grad.detach(), ind=0, percentile=96)
            # plt.figure()
            # plt.imshow(fig_disp)
            # plt.scatter(drawx, drawy, c='g', s=0.5)
            # plt.scatter(sampledx_pts_draw.flatten(), sampledy_pts_draw.flatten(), c='b', s=0.5)
            # plt.show()
            #
            # fig_disp_bin = tensor2disp(disparity_grad_bin.detach(), ind=0, percentile=96)
            # plt.figure()
            # plt.imshow(fig_disp_bin)
            # plt.scatter(drawx, drawy, c='g', s=0.5)
            # plt.scatter(sampledx_pts_draw.flatten(), sampledy_pts_draw.flatten(), c='b', s=0.5)
            # plt.show()
            #
            # toshowind = torch.arange(0, vecs_bin.shape[0])[drawSelector]
            # toshowind = toshowind[torch.LongTensor(1).random_(0, toshowind.shape[0])][0]
            # drawx_s = sampledx_pts[toshowind, :][vecs_bin[toshowind, :] == 1].detach().cpu().numpy()
            # drawy_s = sampledy_pts[toshowind, :][vecs_bin[toshowind, :] == 1].detach().cpu().numpy()
            # fig_seman_bin = tensor2disp(semantics_grad_bin.detach(), ind=0, percentile=96)
            # plt.figure()
            # plt.imshow(fig_seman_bin)
            # plt.scatter(drawx_s, drawy_s, c='g', s=0.5)
            # plt.show()
    def visualization2(self, disparity, semantics):
        disparity_init = disparity.clone()
        semantics_init = semantics.clone()
        disparity = nn.Parameter(disparity, requires_grad=True)
        semantics = nn.Parameter(semantics, requires_grad=True)

        itNum = 5000
        model_optimizer = torch.optim.SGD([disparity, semantics], lr=10)
        for i in range(itNum):
            disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
            semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))

            disparity_grad_bin = disparity_grad > self.depthTh
            semantics_grad_bin = semantics_grad > self.semanTh

            validPos = (self.searchKernel(semantics_grad_bin.float()) > 0) * disparity_grad_bin * self.rangeMask

            sampledx = torch.LongTensor(self.batchSize * self.sampleDense).random_(0, self.width).cuda()
            sampledy = torch.LongTensor(self.batchSize * self.sampleDense).random_(0, self.height).cuda()
            selector1 = validPos[self.channelInd, 0, sampledy, sampledx] == 1

            sampledx_selected = sampledx[selector1]
            sampledy_selected = sampledy[selector1]
            channelInd_selected = self.channelInd[selector1]

            remainedNum = sampledx_selected.shape[0]
            sampledx_pts = sampledx_selected.view(-1, 1).expand(-1, self.repNum) + self.addxx.unsqueeze(0).expand(
                remainedNum, -1)
            sampledy_pts = sampledy_selected.view(-1, 1).expand(-1, self.repNum) + self.addyy.unsqueeze(0).expand(
                remainedNum, -1)
            channelInd_pts = channelInd_selected.view(-1, 1).expand(-1, self.repNum)

            vecs1 = disparity_grad[channelInd_pts, 0, sampledy_pts, sampledx_pts]
            vecs1_bin = disparity_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]
            vecs2 = semantics_grad[channelInd_pts, 0, sampledy_pts, sampledx_pts]
            vecs2_bin = semantics_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]

            bin_diff = torch.abs(vecs1_bin.float() - vecs2_bin.float())
            vec1_dummy = torch.ones_like(vecs1) * self.depthTh * bin_diff
            vec2_dummy = torch.ones_like(vecs2) * self.semanTh * bin_diff

            p1 = (torch.mean(torch.abs(vecs1 - vec1_dummy)) + self.eps).detach()
            p2 = (torch.mean(torch.abs(vecs2 - vec2_dummy)) + self.eps).detach()
            loss = p1 / (p1 + p2) * torch.mean(torch.abs(vecs1 - vec1_dummy)) + p2 / (p1 + p2) * torch.mean(torch.abs(vecs2 - vec2_dummy))
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()
            print("loss is %f" % loss)

            if np.mod(i, 10) == 0:
                fig_disp_grad = tensor2disp(disparity_grad_bin.detach(), ind=0, vmax=1)
                fig_seman_grad = tensor2disp(semantics_grad_bin.detach(), ind=0, vmax=1)
                fig_disp = tensor2disp(disparity.detach(), ind=0, vmax=0.11)
                fig_seman = tensor2disp(semantics, ind=0, vmax=1)
                pil.fromarray(np.concatenate([np.array(fig_disp), np.array(fig_seman), np.array(fig_disp_grad), np.array(fig_seman_grad)], axis=0)).save("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderMerge2/" + str(i) + ".png")

        a = 1
        torch.mean(torch.abs(disparity_init - disparity))
        torch.mean(torch.abs(semantics_init - semantics))
            # vecs_bin = semantics_grad_bin[channelInd_pts, 0, sampledy_pts, sampledx_pts]
            #
            # vecs1 = vecs1 * vecs_bin.float()
            # vecs2 = vecs2 * vecs_bin.float()
            #
            # vecs1 = (vecs1 - torch.mean(vecs1, dim=1, keepdim=True).expand(-1, self.repNum))
            # vecs1 = vecs1 / torch.norm(vecs1, dim=1, keepdim=True).expand(-1, self.repNum)
            # vecs2 = (vecs2 - torch.mean(vecs2, dim=1, keepdim=True).expand(-1, self.repNum))
            # vecs2 = vecs2 / torch.norm(vecs2, dim=1, keepdim=True).expand(-1, self.repNum)

            # loss = torch.mean(torch.sum(vecs1 * vecs2, dim=1))

            # tensor2disp(semantics_grad_bin.detach(), ind=0, vmax=1).show()
            # tensor2disp(disparity_grad.detach(), ind=0, percentile=96).show()
            # tensor2disp(semantics_grad.detach(), ind=0, percentile=96).show()
            # tensor2disp(disparity_grad_bin.detach(), ind=0, vmax=1).show()
            # tensor2disp(validPos.detach(), ind=0, vmax=1).show()
            #
            # viewind = 0
            # drawSelector = channelInd_selected == viewind
            # drawx = sampledx_selected[drawSelector].detach().cpu().numpy()
            # drawy = sampledy_selected[drawSelector].detach().cpu().numpy()
            #
            # sampledx_pts_draw = sampledx_pts[drawSelector, :].detach().cpu().numpy()
            # sampledy_pts_draw = sampledy_pts[drawSelector, :].detach().cpu().numpy()
            # fig_disp = tensor2disp(disparity_grad.detach(), ind=0, percentile=96)
            # plt.figure()
            # plt.imshow(fig_disp)
            # plt.scatter(drawx, drawy, c='g', s=0.5)
            # plt.scatter(sampledx_pts_draw.flatten(), sampledy_pts_draw.flatten(), c='b', s=0.5)
            # plt.show()
            #
            # fig_disp_bin = tensor2disp(disparity_grad_bin.detach(), ind=0, percentile=96)
            # plt.figure()
            # plt.imshow(fig_disp_bin)
            # plt.scatter(drawx, drawy, c='g', s=0.5)
            # plt.scatter(sampledx_pts_draw.flatten(), sampledy_pts_draw.flatten(), c='b', s=0.5)
            # plt.show()
            #
            # toshowind = torch.arange(0, vecs_bin.shape[0])[drawSelector]
            # toshowind = toshowind[torch.LongTensor(1).random_(0, toshowind.shape[0])][0]
            # drawx_s = sampledx_pts[toshowind, :][vecs_bin[toshowind, :] == 1].detach().cpu().numpy()
            # drawy_s = sampledy_pts[toshowind, :][vecs_bin[toshowind, :] == 1].detach().cpu().numpy()
            # fig_seman_bin = tensor2disp(semantics_grad_bin.detach(), ind=0, percentile=96)
            # plt.figure()
            # plt.imshow(fig_seman_bin)
            # plt.scatter(drawx_s, drawy_s, c='g', s=0.5)
            # plt.show()

    def forward3(self, disparity, semantics):
        disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
        semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))

        with torch.no_grad():
            disparity_grad_bin = disparity_grad > self.depthTh
            semantics_grad_bin = semantics_grad > self.semanTh
            joint = (disparity_grad_bin + semantics_grad_bin) > self.eps

            disparity_grad_bin = disparity_grad_bin.float()
            semantics_grad_bin = semantics_grad_bin.float()

            validPos = (self.searchKernel(semantics_grad_bin) > 0) * (self.searchKernel(disparity_grad_bin) > 0) * joint.byte() * self.rangeMask
            # validPos = ((self.searchKernel(semantics_grad_bin) > self.eps) * (
            #             disparity_grad_bin > self.eps) * self.rangeMask)
            combinedVisualization = torch.cat([semantics_grad_bin, validPos.float(), torch.zeros_like(semantics_grad_bin)], dim=1)
            rgb_combined_fig = tensor2rgb(combinedVisualization.detach(), ind=0)

            inited_values_max = self.maxpool(disparity_grad)
            inited_values_min = -self.maxpool(-disparity_grad)

            sxx = self.xx[validPos]
            syy = self.yy[validPos]
            scc = self.channelInd[validPos]
            inited_max = inited_values_max[validPos]
            inited_min = inited_values_min[validPos]
            totNum = sxx.shape[0]

            sxx_expand = sxx.view(-1, 1).expand(-1, self.expandedNum)
            sxx_expand = sxx_expand + self.addxx.view(1, -1).expand(totNum, -1)
            syy_expand = syy.view(-1, 1).expand(-1, self.expandedNum)
            syy_expand = syy_expand + self.addyy.view(1, -1).expand(totNum, -1)
            scc_expand = scc.view(-1, 1).expand(-1, self.expandedNum)
            disp_expand = disparity_grad[scc_expand, 0, syy_expand, sxx_expand]

            center_max = inited_max
            center_min = inited_min
            for k in range(self.itNum):
                center_max = center_max.view(-1, 1).expand(-1, self.expandedNum)
                center_min = center_min.view(-1, 1).expand(-1, self.expandedNum)
                selector = (torch.abs(disp_expand - center_max) <= torch.abs(disp_expand - center_min)).float()
                selector_inv = 1 - selector
                center_max = torch.sum(disp_expand * selector, dim=1) / (torch.sum(selector, dim=1) + self.eps)
                center_min = torch.sum(disp_expand * selector_inv, dim=1) / (torch.sum(selector_inv, dim=1) + self.eps)

            semanBinRe = semantics_grad_bin[validPos]
            dispBinRe = disparity_grad_bin[validPos]
        disparity_grad_vals = disparity_grad[scc, 0, syy, sxx]
        # maxFlow = ((semanBinRe - dispBinRe) > 0).float()
        # minFlow = ((semanBinRe - dispBinRe) < 0).float()
        maxFlow = ((semanBinRe) > 0).float()
        minFlow = 1 - maxFlow
        dst_grad = center_max * semanBinRe.float() + center_min * (1 - semanBinRe).float()
        # dst_grad = semantics_grad[validPos]
        # disparity_grad_vals = self.sig((disparity_grad_vals - self.depthTh))
        # dst_grad = self.sig((dst_grad - self.semanTh))
        # loss = torch.sum(torch.abs(disparity_grad_vals - dst_grad)) / (torch.sum(validPos.float()) + self.eps)


        # if not bool(self.rec):
        #     self.rec['center_max'] = center_max.detach()
        #     self.rec['center_min'] = center_min.detach()
        #     self.rec['validPos'] = validPos.detach()
        # else:
        #     disparity_grad_vals = disparity_grad[self.rec['validPos']]
        #     semanBinRe = semantics_grad_bin[self.rec['validPos']]
        #     dispBinRe = disparity_grad_bin[self.rec['validPos']]
        #     dst_grad = self.rec['center_max'] * semanBinRe.float() + self.rec['center_min'] * (1 - semanBinRe).float()
        #     maxFlow = ((semanBinRe) > 0).float()
        #     minFlow = 1 - maxFlow
        #     print("Diff value is %f" % torch.sum(torch.abs(disparity_grad_vals - dst_grad)))

        # loss = torch.sum(torch.abs(disparity_grad_vals-dst_grad) * (torch.abs(dispBinRe - semanBinRe).float())) / (torch.sum(torch.abs(dispBinRe - semanBinRe).float()) + self.eps)
        # loss = torch.sum(torch.abs(disparity_grad_vals - center_max) * maxFlow) / torch.sum(maxFlow + self.eps)
        # print("v1 % f" % torch.sum(torch.abs(disparity_grad_vals - dst_grad) * minFlow) / torch.sum(minFlow + self.eps))
        # print("v2 % f" % torch.sum(torch.abs(disparity_grad_vals - dst_grad) * maxFlow) / torch.sum(maxFlow + self.eps))
        # loss = torch.sum(torch.abs(disparity_grad_vals - dst_grad) * minFlow) / torch.sum(minFlow + self.eps) \
        #         + torch.sum(torch.abs(disparity_grad_vals - dst_grad) * maxFlow) / torch.sum(maxFlow + self.eps)
        a1 = torch.sum(torch.abs(disparity_grad_vals - dst_grad) * minFlow) / torch.sum(minFlow + self.eps)
        a2 = torch.sum(torch.abs(disparity_grad_vals - dst_grad) * maxFlow) / torch.sum(maxFlow + self.eps)
        loss = a1 + a2
        # print("a1 % f" % a1)
        # print("a2 % f" % a2)

        return loss
    def visualization3(self, disparity, semantics):
        disparity_init = disparity.clone()
        semantics_init = semantics.clone()
        disparity = nn.Parameter(disparity, requires_grad=True)
        semantics = nn.Parameter(semantics, requires_grad=True)

        itNum = 5000
        model_optimizer = torch.optim.SGD([disparity, semantics], lr=10)
        for i in range(itNum):
            disparity_grad = torch.abs(self.convDispx(disparity)) + torch.abs(self.convDispy(disparity))
            semantics_grad = torch.abs(self.convDispx(semantics)) + torch.abs(self.convDispy(semantics))

            disparity_grad_bin = disparity_grad > self.depthTh
            semantics_grad_bin = semantics_grad > self.semanTh

            disparity_grad_bin = disparity_grad_bin.float()
            semantics_grad_bin = semantics_grad_bin.float()

            validPos = (self.searchKernel(semantics_grad_bin) > 0) * disparity_grad_bin.byte() * self.rangeMask
            combinedVisualization = torch.cat([semantics_grad_bin, validPos.float(), torch.zeros_like(semantics_grad_bin)], dim=1)
            rgb_combined_fig = tensor2rgb(combinedVisualization.detach(), ind=0)
            # tensor2disp(torch.abs(disparity_grad_sig - semantics_grad_sig), ind=0, vmax=1).show()
            # tensor2disp(disparity_grad_sig, ind=0, vmax=0.6).show()

            inited_values_max = self.maxpool(disparity_grad)
            inited_values_min = -self.maxpool(-disparity_grad)

            sxx = self.xx[validPos]
            syy = self.yy[validPos]
            scc = self.channelInd[validPos]
            inited_max = inited_values_max[validPos]
            inited_min = inited_values_min[validPos]
            totNum = sxx.shape[0]

            sxx_expand = sxx.view(-1, 1).expand(-1, self.expandedNum)
            sxx_expand = sxx_expand + self.addxx.view(1, -1).expand(totNum, -1)
            syy_expand = syy.view(-1, 1).expand(-1, self.expandedNum)
            syy_expand = syy_expand + self.addyy.view(1, -1).expand(totNum, -1)
            scc_expand = scc.view(-1, 1).expand(-1, self.expandedNum)
            disp_expand = disparity_grad[scc_expand, 0, syy_expand, sxx_expand]

            # center_max = inited_max.view(-1, 1).expand(-1, self.expandedNum)
            # center_min = inited_min.view(-1, 1).expand(-1, self.expandedNum)
            center_max = inited_max
            center_min = inited_min
            for k in range(self.itNum):
                center_max = center_max.view(-1, 1).expand(-1, self.expandedNum)
                center_min = center_min.view(-1, 1).expand(-1, self.expandedNum)
                selector = (torch.abs(disp_expand - center_max) <= torch.abs(disp_expand - center_min)).float()
                selector_inv = 1 - selector
                center_max = torch.sum(disp_expand * selector, dim=1) / (torch.sum(selector, dim=1) + self.eps)
                center_min = torch.sum(disp_expand * selector_inv, dim=1) / (torch.sum(selector_inv, dim=1) + self.eps)

            semanBinRe = semantics_grad_bin[validPos]
            dispBinRe = disparity_grad_bin[validPos]
            disparity_grad_vals = disparity_grad[scc, 0, syy, sxx]

            maxFlow = ((semanBinRe - dispBinRe) > 0).float()
            minFlow = ((semanBinRe - dispBinRe) < 0).float()

            loss = torch.sum(torch.abs(disparity_grad_vals - center_min) * minFlow + torch.abs(disparity_grad_vals - center_max) * maxFlow) / torch.sum(minFlow + maxFlow + self.eps)
            model_optimizer.zero_grad()
            loss.backward()
            model_optimizer.step()



            plt.figure()
            plt.imshow(rgb_combined_fig)
            viewInd = 0
            viewSelector = scc_expand == viewInd
            show_xx = sxx_expand[viewSelector].detach().cpu().numpy()
            show_yy = syy_expand[viewSelector].detach().cpu().numpy()
            plt.scatter(show_xx, show_yy, s = 0.1, c = 'b')
            plt.show()

            if np.mod(i, 10) == 0:
                fig_disp_grad = tensor2disp(disparity_grad_bin.detach(), ind=0, vmax=1)
                fig_seman_grad = tensor2disp(semantics_grad_bin.detach(), ind=0, vmax=1)
                fig_disp = tensor2disp(disparity.detach(), ind=0, vmax=0.11)
                fig_seman = tensor2disp(semantics, ind=0, vmax=1)
                pil.fromarray(np.concatenate(
                    [np.array(fig_disp),
                     np.array(fig_seman),
                     np.array(fig_disp_grad),
                     np.array(fig_seman_grad)], axis=0)).save("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderMerge2/" + str(i) + ".png")




class ComputeSurroundingPixDistance(nn.Module):
    def __init__(self, height, width, batch_size, intrinsic_set, extrinsic_set, kernel_size = 3):
        super(ComputeSurroundingPixDistance, self).__init__()
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.intrinsic_set = intrinsic_set
        self.extrinsic_set = extrinsic_set
        self.kernel_size = kernel_size
        self.A_set = dict()
        for key in self.intrinsic_set:
            self.A_set[key[:-3]] = torch.inverse(self.intrinsic_set[key] @ self.extrinsic_set[key])

        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)
        self.xx = torch.from_numpy(xx).cuda()
        self.yy = torch.from_numpy(yy).cuda()
        self.compare_num = kernel_size * kernel_size - 1

        subTractweight = nn.Parameter(torch.zeros([self.kernel_size * self.kernel_size - 1, 1, self.kernel_size, self.kernel_size]), requires_grad=False)
        selectWeight = nn.Parameter(torch.zeros([self.kernel_size * self.kernel_size - 1, 1, self.kernel_size, self.kernel_size]), requires_grad=False)
        count = 0
        for i in range(0, self.kernel_size):
            for j in range(0, self.kernel_size):
                if i == int((self.kernel_size - 1) / 2) and j == int((self.kernel_size - 1) / 2):
                    continue
                subTractweight[count, 0, i, j] = 1
                subTractweight[count, 0, int((self.kernel_size - 1) / 2), int((self.kernel_size - 1) / 2)] = -1
                selectWeight[count, 0, int((self.kernel_size - 1) / 2), int((self.kernel_size - 1) / 2)] = 1
                count = count + 1
        self.subtracion_counter = torch.nn.Conv2d(in_channels=1, out_channels=self.kernel_size * self.kernel_size - 1, kernel_size=self.kernel_size, bias=False, padding=int((self.kernel_size - 1) / 2))
        self.subtracion_counter.weight = subTractweight
        self.subtracion_counter.cuda()

        self.selection_counter = torch.nn.Conv2d(in_channels=1, out_channels=self.kernel_size * self.kernel_size - 1, kernel_size=self.kernel_size, bias=False, padding=int((self.kernel_size - 1) / 2))
        self.selection_counter.weight = selectWeight
        self.selection_counter.cuda()
        # self.pix_coords = torch.from_numpy(np.stack([xx, yy], axis=0))
        # xx = xx.flatten().astype(np.float32)
        # yy = yy.flatten().astype(np.float32)
        # self.pix_coords = np.expand_dims(np.stack([xx, yy, np.ones(self.width * self.height).astype(np.float32)], axis=1), axis=0).repeat(self.batch_size, axis=0)
        # self.pix_coords = torch.from_numpy(self.pix_coords).permute(0,2,1)
        # self.ones = torch.ones(self.batch_size, 1, self.height * self.width)
        # self.pix_coords = self.pix_coords.cuda()
        # self.ones = self.ones.cuda()

        self.dirxset = list()
        self.biasxset = list()
        self.diryset = list()
        self.biasyset = list()
        self.dirzset = list()
        self.biaszset = list()
        self.translator = dict()
        count = 0
        for key in self.intrinsic_set:
            self.translator[key[:-3]] = count
            self.dirxset.append( self.xx * self.A_set[key[:-3]][0, 0] + self.yy * self.A_set[key[:-3]][0, 1] + self.A_set[key[:-3]][0, 2])
            self.biasxset.append(torch.ones([self.height, self.width], device=torch.device("cuda")) * self.A_set[key[:-3]][0, 3])

            self.diryset.append( self.xx * self.A_set[key[:-3]][1, 0] + self.yy * self.A_set[key[:-3]][1, 1] + self.A_set[key[:-3]][1, 2])
            self.biasyset.append(torch.ones([self.height, self.width], device=torch.device("cuda")) * self.A_set[key[:-3]][1, 3])

            self.dirzset.append( self.xx * self.A_set[key[:-3]][2, 0] + self.yy * self.A_set[key[:-3]][2, 1] + self.A_set[key[:-3]][2, 2])
            self.biaszset.append(torch.ones([self.height, self.width], device=torch.device("cuda")) * self.A_set[key[:-3]][2, 3])
            count = count + 1
        self.dirxset = torch.stack(self.dirxset, dim=0).unsqueeze(1)
        self.biasxset = torch.stack(self.biasxset, dim=0).unsqueeze(1)
        self.diryset = torch.stack(self.diryset, dim=0).unsqueeze(1)
        self.biasyset = torch.stack(self.biasyset, dim=0).unsqueeze(1)
        self.dirzset = torch.stack(self.dirzset, dim=0).unsqueeze(1)
        self.biaszset = torch.stack(self.biaszset, dim=0).unsqueeze(1)

        self.surroundxdir = list()
        self.surroundxbias = list()
        self.surroundydir = list()
        self.surroundybias = list()
        self.surroundzdir = list()
        self.surroundzbias = list()

        for i in range(self.dirxset.shape[0]):
            self.surroundxdir.append(self.selection_counter(self.dirxset[i, :, :, :].unsqueeze(0)))
            self.surroundxbias.append(self.selection_counter(self.biasxset[i, :, :, :].unsqueeze(0)))
            self.surroundydir.append(self.selection_counter(self.diryset[i, :, :, :].unsqueeze(0)))
            self.surroundybias.append(self.selection_counter(self.biasyset[i, :, :, :].unsqueeze(0)))
            self.surroundzdir.append(self.selection_counter(self.dirzset[i, :, :, :].unsqueeze(0)))
            self.surroundzbias.append(self.selection_counter(self.biaszset[i, :, :, :].unsqueeze(0)))


        self.surroundxdir = torch.cat(self.surroundxdir, dim=0)
        self.surroundxbias = torch.cat(self.surroundxbias, dim=0)
        self.surroundydir = torch.cat(self.surroundydir, dim=0)
        self.surroundybias = torch.cat(self.surroundybias, dim=0)
        self.surroundzdir = torch.cat(self.surroundzdir, dim=0)
        self.surroundzbias = torch.cat(self.surroundzbias, dim=0)


        normval = (torch.sqrt(self.surroundxdir * self.surroundxdir + self.surroundydir * self.surroundydir + self.surroundzdir * self.surroundzdir) + 1e-6)
        self.surroundxdir = self.surroundxdir / normval
        self.surroundydir = self.surroundydir / normval
        self.surroundzdir = self.surroundzdir / normval


    def acquire_key(self, intrinsic, extrinsic):
        key_val_list = list((torch.sum(torch.sum(intrinsic, dim=1), dim=1) + torch.sum(torch.sum(extrinsic, dim=1), dim=1)).cpu().numpy())
        for idx, val in enumerate(key_val_list):
            key_val_list[idx] = str(val)[:-3]

        ind_list = list()
        for ky in key_val_list:
            ind_list.append(self.translator[ky])
        ind_list = torch.Tensor(ind_list).long().cuda()
        return key_val_list, ind_list

    def get_3d_pts(self, depthmap, intrinsic, extrinsic):
        kys, inds = self.acquire_key(intrinsic, extrinsic)
        pts_3dx = depthmap * self.dirxset[inds, :, :, :] + self.biasxset[inds, :, :, :]
        pts_3dy = depthmap * self.diryset[inds, :, :, :] + self.biasyset[inds, :, :, :]
        pts_3dz = depthmap * self.dirzset[inds, :, :, :] + self.biaszset[inds, :, :, :]
        sub_in_x = self.subtracion_counter(pts_3dx)
        sub_in_y = self.subtracion_counter(pts_3dy)
        sub_in_z = self.subtracion_counter(pts_3dz)
        min_val, min_ind = torch.min(torch.sqrt(sub_in_x * sub_in_x + sub_in_y * sub_in_y + sub_in_z * sub_in_z + 1e-15), dim=1, keepdim=True)
        tmpx = (self.surroundxbias[inds, :, :, :] - pts_3dx.expand(-1,self.compare_num,-1,-1)) * self.surroundxdir[inds, :, :]
        tmpy = (self.surroundybias[inds, :, :, :] - pts_3dy.expand(-1, self.compare_num, -1, -1)) * self.surroundydir[inds, :, :]
        tmpz = (self.surroundzbias[inds, :, :, :] - pts_3dz.expand(-1, self.compare_num, -1, -1)) * self.surroundzdir[inds, :, :]
        distance_along_ray = torch.sqrt(tmpx * tmpx + tmpy * tmpy + tmpz * tmpz)
        ttx = pts_3dx.expand(-1,self.compare_num,-1,-1) - (self.surroundxbias[inds, :, :, :] + distance_along_ray * self.surroundxdir[inds, :, :, :])
        tty = pts_3dy.expand(-1, self.compare_num, -1, -1) - (self.surroundybias[inds, :, :, :] + distance_along_ray * self.surroundydir[inds, :, :, :])
        ttz = pts_3dz.expand(-1, self.compare_num, -1, -1) - (self.surroundzbias[inds, :, :, :] + distance_along_ray * self.surroundzdir[inds, :, :, :])
        tt_dist = torch.sqrt(ttx * ttx + tty * tty + ttz * ttz + 1e-5)
        gathered_distance = torch.gather(tt_dist, 1, min_ind)
        bad_pts = min_val > gathered_distance
        tt_dist_mean = torch.mean(tt_dist, dim=1, keepdim=True)
        penalty = torch.sum(tt_dist_mean * bad_pts.float()) / (torch.sum(bad_pts) + 1)
        return bad_pts


    def get_3d_pts_analyze(self, depthmap, intrinsic, extrinsic, semantic = None):
        kys, inds = self.acquire_key(intrinsic, extrinsic)

        pts_3dx = depthmap * self.dirxset[inds, :, :, :] + self.biasxset[inds, :, :, :]
        pts_3dy = depthmap * self.diryset[inds, :, :, :] + self.biasyset[inds, :, :, :]
        pts_3dz = depthmap * self.dirzset[inds, :, :, :] + self.biaszset[inds, :, :, :]

        # p_ms = list()
        # for ky in kys:
        #     p_ms.append(self.A_set[ky])
        # p_ms = torch.stack(p_ms, dim=0).cuda()
        #
        # pts_3dx = \
        #             self.xx * p_ms[:, 0, 0].view(self.batch_size,1,1,1).expand(-1,-1,self.height, self.width) * depthmap + \
        #             self.yy * p_ms[:, 0, 1].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             p_ms[:, 0, 2].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             p_ms[:, 0, 3].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width)
        # pts_3dy = \
        #             self.xx * p_ms[:, 1, 0].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             self.yy * p_ms[:, 1, 1].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             p_ms[:, 1, 2].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             p_ms[:, 1, 3].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width)
        # pts_3dz = \
        #             self.xx * p_ms[:, 2, 0].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             self.yy * p_ms[:, 2, 1].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             p_ms[:, 2, 2].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width) * depthmap + \
        #             p_ms[:, 2, 3].view(self.batch_size, 1, 1, 1).expand(-1, -1, self.height, self.width)
        #
        sub_in_x = self.subtracion_counter(pts_3dx)
        sub_in_y = self.subtracion_counter(pts_3dy)
        sub_in_z = self.subtracion_counter(pts_3dz)
        min_val, min_ind = torch.min(torch.sqrt(sub_in_x * sub_in_x + sub_in_y * sub_in_y + sub_in_z * sub_in_z + 1e-15), dim=1, keepdim=True)

        tmpx = (self.surroundxbias[inds, :, :, :] - pts_3dx.expand(-1,self.compare_num,-1,-1)) * self.surroundxdir[inds, :, :]
        tmpy = (self.surroundybias[inds, :, :, :] - pts_3dy.expand(-1, self.compare_num, -1, -1)) * self.surroundydir[inds, :, :]
        tmpz = (self.surroundzbias[inds, :, :, :] - pts_3dz.expand(-1, self.compare_num, -1, -1)) * self.surroundzdir[inds, :, :]
        distance_along_ray = torch.sqrt(tmpx * tmpx + tmpy * tmpy + tmpz * tmpz)
        ttx = pts_3dx.expand(-1,self.compare_num,-1,-1) - (self.surroundxbias[inds, :, :, :] + distance_along_ray * self.surroundxdir[inds, :, :, :])
        tty = pts_3dy.expand(-1, self.compare_num, -1, -1) - (
                    self.surroundybias[inds, :, :, :] + distance_along_ray * self.surroundydir[inds, :, :, :])
        ttz = pts_3dz.expand(-1, self.compare_num, -1, -1) - (
                    self.surroundzbias[inds, :, :, :] + distance_along_ray * self.surroundzdir[inds, :, :, :])
        tt_dist = torch.sqrt(ttx * ttx + tty * tty + ttz * ttz + 1e-5)
        gathered_distance = torch.gather(tt_dist, 1, min_ind, out=None, sparse_grad=False)
        bad_pts = min_val > gathered_distance


        viewInd = 0
        seman_fig = tensor2semantic(semantic, ind=viewInd)
        seman_color = np.array(seman_fig).reshape(-1, 3).astype(np.float) / 255
        semantic_curview = semantic[viewInd, 0, :, :].detach().cpu().numpy().flatten()

        sx = pts_3dx[viewInd, 0, :, :].detach().cpu().numpy().flatten()
        sy = pts_3dy[viewInd, 0, :, :].detach().cpu().numpy().flatten()
        sz = pts_3dz[viewInd, 0, :, :].detach().cpu().numpy().flatten()

        bad_pts_selection = (bad_pts[viewInd, 0, :, :].detach().cpu().numpy().flatten() == 1)
        bsx = sx[bad_pts_selection]
        bsy = sy[bad_pts_selection]
        bsz = sz[bad_pts_selection]
        bad_seman = semantic_curview[bad_pts_selection]
        for lab in labels:
            if lab.trainId != 255:
                print("Cat %s percentage is %f." % (lab.name, np.sum(bad_seman == lab.trainId) / bad_seman.shape[0]))

        selector = np.sqrt(sx*sx + sy * sy + sz * sz) < 5
        sx = sx[selector]
        sy = sy[selector]
        sz = sz[selector]
        seman_color = seman_color[selector, :]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(sx[0::50], sy[0::50], sz[0::50], c = seman_color[0::50, :], s=0.1)
        ax.scatter(sx[0::50], sy[0::50], sz[0::50], c='b', s=0.1)
        ax.scatter(bsx, bsy, bsz, c='r', s=0.5)
        set_axes_equal(ax)


class Subpixel_metric(nn.Module):
    def __init__(self):
        super(Subpixel_metric, self).__init__()
        self.contrast_left = torch.nn.Conv2d(in_channels=3, out_channels=3, groups=3, kernel_size=(1, 3),
                                             padding=(0, 1), bias=False)
        self.contrast_right = torch.nn.Conv2d(in_channels=3, out_channels=3, groups=3, kernel_size=(1, 3),
                                              padding=(0, 1), bias=False)

        left_weight = torch.zeros((3, 1, 1, 3))
        right_weight = torch.zeros((3, 1, 1, 3))
        left_weight[:, 0, 0, 0:2] = 0.5
        right_weight[:, 0, 0, 1:3] = 0.5
        self.contrast_left.weight = nn.Parameter(left_weight, requires_grad=False)
        self.contrast_right.weight = nn.Parameter(right_weight, requires_grad=False)
    def forward(self, predict, target):
        sub_pixell = self.contrast_left(target)
        sub_pixelr = self.contrast_right(target)

        # tensor2rgb(sub_pixell, ind=0).show()
        contrast_vals = torch.stack(
            [torch.abs(predict- sub_pixell), torch.abs(predict - target), torch.abs(predict - sub_pixelr)], dim=1)
        contrast_vals, _ = torch.min(contrast_vals, dim=1)
        return torch.mean(contrast_vals, dim=1, keepdim=True)