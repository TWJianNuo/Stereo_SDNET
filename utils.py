# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
import numpy as np
import torch
from six.moves import urllib
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
from globalInfo import acGInfo
from torch.utils.data.sampler import Sampler
from random import shuffle
import PIL.Image as pil
from cityscapesscripts.helpers.labels import *
import math
import itertools
# from numba import jit



def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def reconstruct3dPts(depthval, ix, iy, intrinsic, extrinsic):
    recon3Pts = np.stack((ix * depthval, iy * depthval, depthval, depthval / depthval), axis=1)
    recon3Pts = (np.linalg.inv(intrinsic @ extrinsic) @ recon3Pts.T).T
    return recon3Pts

def project3dPts(pts3d, intrinsic, extrinsic, isDepth = False):
    pts2d = (intrinsic @ extrinsic @ pts3d.T).T
    pts2d[:, 0] = pts2d[:, 0] / pts2d[:, 2]
    pts2d[:, 1] = pts2d[:, 1] / pts2d[:, 2]
    depth = pts2d[:, 2]
    pts2d = pts2d[:, 0:2]
    if not isDepth:
        return pts2d
    else:
        return pts2d, depth
def diffICP(depthMap, pts3d, intrinsic, extrinsic, svInd = 0):
    # This is a function doing differentiable ICP loss
    svAddPre = acGInfo()['internalRe_add']
    svAdd = os.path.join(svAddPre, 'icpLoss')
    pts3d_org = np.copy(pts3d)
    iterationTime = 5
    affM = np.eye(4)
    distTh = 0.6
    icpLoss = np.zeros(iterationTime)
    matchedPtsRec = np.zeros(iterationTime)
    successTime = 0
    for k in range(iterationTime):
        pts3d = (affM @ pts3d.T).T
        pts2d, depth = project3dPts(pts3d, intrinsic, extrinsic, isDepth = True)
        imgShape = depthMap.shape
        interX = np.arange(0, imgShape[1], 1)
        interY = np.arange(0, imgShape[0], 1)
        interpF = RectBivariateSpline(interY, interX, depthMap)
        depthR = interpF.ev(pts2d[:, 1], pts2d[:, 0])
        pts3dR = reconstruct3dPts(depthR, pts2d[:, 0], pts2d[:, 1], intrinsic, extrinsic)
        validMask = np.sqrt(np.sum(np.square(pts3d - pts3dR), axis=1)) < distTh
        if np.sum(validMask) <= 30:
            break
        valpts3dR = pts3dR[validMask, :]
        valpts3d = pts3d[validMask, :]
        icpLoss[k] = np.mean(np.sum(np.square(pts3d[validMask, :] - pts3dR[validMask, :]), axis=1))
        matchedPtsRec[k] = np.sum(validMask)
        affM = valpts3dR.T @ valpts3d @ np.linalg.inv(valpts3d.T @ valpts3d)
        successTime = successTime + 1
    if successTime == 0:
        fig = plt.figure()
        ax = fig.add_subplot((111), projection='3d')
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], s=0.1, c='r')
        ax.scatter(pts3dR[:, 0], pts3dR[:, 1], pts3dR[:, 2], s=0.1, c='b')
        set_axes_equal(ax)
        plt.legend(['org Pts', 'target pts'])
        plt.title("Failure case")
        plt.savefig(os.path.join(svAdd, str(svInd) + '_3d'), dpi=300)
        plt.close(fig)

        fig = plt.figure()
        plt.stem(icpLoss)
        plt.savefig(os.path.join(svAdd, str(svInd) + '_curve'))
        plt.close(fig)
        return None, None
    fig = plt.figure()
    ax = fig.add_subplot((111), projection='3d')
    ax.scatter(pts3d[validMask, 0], pts3d[validMask, 1], pts3d[validMask, 2], s=0.1, c='r')
    ax.scatter(pts3dR[validMask, 0], pts3dR[validMask, 1], pts3dR[validMask, 2], s=0.1, c='b')
    ax.scatter(pts3d_org[validMask, 0], pts3d_org[validMask, 1], pts3d_org[validMask, 2], s=0.1, c='g')
    set_axes_equal(ax)
    plt.legend(['src Pts', 'target pts', 'org pts'])
    plt.savefig(os.path.join(svAdd, str(svInd) + '_3d'), dpi=300)
    plt.close(fig)
    # fig.show()

    occptRat = matchedPtsRec[successTime - 1] / matchedPtsRec[0]
    fig = plt.figure()
    plt.stem(icpLoss)
    plt.title("ICP loss curve, occupancy Ratio is %f" % occptRat)
    plt.xlabel("Iteration times")
    plt.ylabel("Square loss")
    plt.savefig(os.path.join(svAdd, str(svInd) + '_curve'))
    plt.close(fig)
    return validMask, affM
    # plt.show()

class my_Sampler(Sampler):
    def __init__(self, nums, batch_num):
        self.batch_num = batch_num
        self.nums = nums
        # init iter
        self.iter_list = list()
        for i, num in enumerate(self.nums):
            bot_num = np.sum(self.nums[0:i])
            top_num = np.sum(self.nums[0:i+1])
            indices = list(range(bot_num, top_num))
            shuffle(indices)
            for k in range(np.int(np.floor(len(indices) / self.batch_num))):
                self.iter_list.append(indices[k * self.batch_num : (k+1) * self.batch_num])
        self.num_samples = self.batch_num * len(self.iter_list)

    def __iter__(self):
        # shuffle when called
        shuffle(self.iter_list)
        iterIndex = list()
        for i in self.iter_list:
            iterIndex = iterIndex + i
        assert len(iterIndex) == np.unique(np.array(iterIndex)).shape[0], "iter error"
        # print("called with len %d" % self.nums)
        # print ('\tcalling Sampler:__iter__')
        return iter(iterIndex )

    def __len__(self):
        # print ('\tcalling Sampler:__len__')
        return self.num_samples

def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

class my_Sampler_group2(Sampler):
    def __init__(self, nums, batch_num):
        self.batch_num = batch_num
        self.nums = nums
        # init iter
        self.iter_list = list(np.arange(nums))
        self.num_samples = nums

    def __iter__(self):
        # shuffle when called
        ind_arr = np.arange(len(self.iter_list))
        ind_arr = ind_arr.reshape(int(len(self.iter_list) / 2), 2)
        np.random.shuffle(ind_arr)
        ind_arr = list(ind_arr.flatten().astype(np.int))
        return iter(ind_arr)

    def __len__(self):
        # print ('\tcalling Sampler:__len__')
        return self.num_samples


def visualize_outpu(inputs, outputs, sv_path, sv_ind):
    for picind in range(inputs[('color', 0, 0)].shape[0]):
        c_sv_path = os.path.join(sv_path, str(sv_ind) + "_" + str(picind) + ".png")
        img1 = inputs[('color', 's', 0)].permute(0, 2, 3, 1)[picind, :, :, :].clone().detach().cpu().numpy()
        img2 = outputs[('color', 's', 0)].permute(0, 2, 3, 1)[picind, :, :, :].clone().detach().cpu().numpy()
        img3 = inputs[('color', 0, 0)].permute(0, 2, 3, 1)[picind, :, :, :].clone().detach().cpu().numpy()
        combined_img = np.concatenate((img1, img2, img3), axis=0)
        pil.fromarray((combined_img * 255).astype(np.uint8)).save(c_sv_path)

def visualize_semantic(img_inds):
    # please input numpy array
    size = [img_inds.shape[1], img_inds.shape[0]]
    background = name2label['unlabeled'].color
    labelImg = np.array(pil.new("RGB", size, background))
    for id in trainId2label.keys():
        if id >= 0:
            label = trainId2label[id].name
        else:
            label = 'unlabeled'
        color = name2label[label].color
        mask = img_inds == id
        labelImg[mask, :] = color
    return pil.fromarray(labelImg)
    # labelImg = pil.fromarray(labelImg).show()
def visualize_rgbTensor(pred, view_ind = 0):
    pred = pred.permute(0,2,3,1)
    pred = (pred[view_ind, :, :, :].detach().cpu().numpy() * 255).astype(np.uint8)
    pil.fromarray(pred).show()

def con_sin(angle):
    return math.cos(angle), math.sin(angle)

def angle2matrix(pitch, roll, yaw):
    cy, sy = con_sin(yaw)
    cr, sr = con_sin(roll)
    cp, sp = con_sin(pitch)
    ex = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr]
    ]
    return np.array(ex)

def get_gaussian_kernel_weights(kernel_size=3, sigma=2):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    # gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    #
    # gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
    #                             kernel_size=kernel_size, groups=channels, bias=False, padding = padding)
    #
    # gaussian_filter.weight.data = gaussian_kernel
    # gaussian_filter.weight.requires_grad = False

    return gaussian_kernel


def tensor2rgb(tensor, ind):
    slice = (tensor[ind, :, :, :].permute(1,2,0).detach().contiguous().cpu().numpy() * 255).astype(np.uint8)
    return pil.fromarray(slice)

def tensor2semantic(tensor, ind, isGt = False):
    slice = tensor[ind, :, :, :]
    slice = slice[0,:,:].detach().cpu().numpy()
    # visualize_semantic(slice).show()
    return visualize_semantic(slice)

def tensor2disp(tensor, ind, vmax = None, percentile = None):
    slice = tensor[ind, 0, :, :].detach().cpu().numpy()
    if percentile is None:
        percentile = 90
    if vmax is None:
        vmax = np.percentile(slice, percentile)
    slice = slice / vmax
    cm = plt.get_cmap('magma')
    slice = (cm(slice) * 255).astype(np.uint8)
    return pil.fromarray(slice[:,:,0:3])

def visualizeNpDisp(arr, vmax = None, percentile = None):
    slice = arr
    if percentile is None:
        percentile = 90
    if vmax is None:
        vmax = np.percentile(slice, percentile)
    slice = slice / vmax
    cm = plt.get_cmap('magma')
    slice = (cm(slice) * 255).astype(np.uint8)
    slice = slice[:,:,0:3]
    return pil.fromarray(slice)

def float2uint8(img):
    return (img * 4.3 * 255).astype(np.uint8) # 4.5 - 0.011182, 5.6 -0.020146, 5.2 - 0.015210, 5.0 - 0.013278, 4.6 - 0.011309

def uint82float(img):
    return (img).astype(np.float) / 4.3 / 255 # 4.5 - 0.011182, 5.6 -0.020146, 5.2 - 0.015210, 5.0 - 0.013278, 4.6 - 0.011309

def visualizeGrad(gradtensor, percentile = 95, viewind = 0):
    cm = plt.get_cmap('bwr')
    gradnumpy = gradtensor.detach().cpu().numpy()[viewind, 0, :, :]

    selector_pos = gradnumpy > 0
    if np.sum(selector_pos) > 1:
        pos_bar = np.percentile(gradnumpy[selector_pos], percentile)
        gradnumpy[selector_pos] = gradnumpy[selector_pos] / pos_bar / 2


    selector_neg = gradnumpy < 0
    if np.sum(selector_neg) > 1:
        neg_bar = -np.percentile(-gradnumpy[selector_neg], percentile)
        gradnumpy[selector_neg] = -gradnumpy[selector_neg] / neg_bar / 2

    disp_grad_numpy = gradnumpy + 0.5
    colorMap = cm(disp_grad_numpy)[:,:,0:3]
    return pil.fromarray((colorMap * 255).astype(np.uint8))

def combined_2_img(img1, img2, rat1):
    img1 = np.array(img1)[:,:,0:3]
    img2 = np.array(img2)[:, :, 0:3]
    img_combined = (img1 * rat1 + img2 * (1 - rat1)).astype(np.uint8)
    return pil.fromarray(img_combined )

def concat_imgs(imgs, dim = 0):
    img_list = list()
    for img in imgs:
        img_list.append(np.array(img)[:,:,0:3])
    return pil.fromarray(np.concatenate(img_list, axis=dim))