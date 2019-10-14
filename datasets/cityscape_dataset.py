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
import torch

from kitti_utils import generate_depth_map
from .SingleDataset import SingleDataset
from torchvision import transforms
import json
import copy
from utils import angle2matrix
from utils import set_axes_equal
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class CITYSCAPEDataset(SingleDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, data_path, filenames, height, width, frame_idxs, num_scales, tag, is_train=False, img_ext='.png', load_depth = False, load_meta = False, is_load_semantics = True):
        super(CITYSCAPEDataset, self).__init__(data_path, filenames, height, width, frame_idxs, num_scales, tag, is_train, img_ext='.png')
        # self.kitti_K = np.array([[0.58, 0, 0.5, 0],
        #                         [0, 1.92, 0.5, 0],
        #                         [0, 0, 1, 0],
        #                         [0, 0, 0, 1]], dtype=np.float32)
        self.downR = 32 # downsample ratio between input img and feature map
        self.kitti_uf = np.array((0.58, 1.92)) # [kitti unit focal x per pixel, kitti unit focal y per pixel]
        self.full_res_shape = (2048, 1024) # decide to use 512 by 256
        self.side_map = {"l": "leftImg8bit", "r": "rightImg8bit"}
        self.change_resize()
        self.mask = None
        self.load_meta = load_meta
        self.load_mask()
        self.load_seman = is_load_semantics
        if load_depth:
            # Need to overwrite prev flag
            self.load_depth = True

    def load_mask(self):
        imgPathl = 'assets/cityscapemask_left.png'
        maskl = dict()
        maskImg = self.loader(imgPathl)
        for i in range(self.num_scales):
            mask = np.array(maskImg.resize((int(self.width / (2**i)), int(self.height / (2**i))), pil.NEAREST))[:,:,0]
            mask = (mask < 1).astype(np.uint8)
            maskl[('mask', i)] = torch.from_numpy(mask)

        imgPathr = 'assets/cityscapemask_right.png'
        maskr = dict()
        maskImg = self.loader(imgPathr)
        for i in range(self.num_scales):
            mask = np.array(maskImg.resize((int(self.width / (2 ** i)), int(self.height / (2 ** i))), pil.NEAREST))[:,:, 0]
            mask = (mask < 1).astype(np.uint8)
            maskr[('mask', i)] = torch.from_numpy(mask)

        self.mask = dict()
        self.mask['left'] = maskl
        self.mask['right'] = maskr
    def check_depth(self):
        return False

    def check_cityscape_meta(self):
        return self.load_meta

    def get_color(self, folder, frame_index, side, do_flip):
        imgPath = self.get_image_path(folder, frame_index, side)
        color = self.loader(imgPath)
        # cts_focal, baseline = self.get_cityscape_cam_param(folder)

        # targ_f = self.kitti_uf * np.array((self.width, self.height))
        # re_size = np.round(self.full_res_shape / cts_focal * targ_f).astype(np.intc)
        # re_size = (np.round(re_size / self.downR) * self.downR).astype(np.intc)
        # self.ctsImg_sz_rec[imgPath] = re_size
        # re_size = self.ctsImg_sz_rec[self.t_folder(folder)]
        # color = color.resize(re_size, self.interp)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color
    def change_resize(self):
        # cts_focal = np.array((2268.36, 2225.5405988775956)) # suppose all cityscape img possess same pixel focal length
        # targ_f = self.kitti_uf * np.array((self.width, self.height))
        # re_size = np.round(self.full_res_shape / cts_focal * targ_f).astype(np.int)
        # re_size = (np.round(re_size / self.downR) * self.downR).astype(np.int)
        # re_size = np.array((512, 256),dtype=np.int)
        # self.org_width = copy.copy(self.width)
        # self.width = re_size[0].item()
        # self.height = re_size[1].item()
        self.width = 512
        self.height = 256

        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)
        self.seman_resize = transforms.Resize((self.height, self.width),
                                               interpolation=pil.NEAREST)

    def get_K(self, folder):
        # Return K as well as record all necessary things
        cts_focal, baseline = self.get_cityscape_cam_param(folder)
        unit_focal_pixel = cts_focal / self.full_res_shape
        K = np.eye(4, dtype=np.float32)
        K[0, 2] = 0.5
        K[1, 2] = 0.5
        K[0, 0] = unit_focal_pixel[0]
        K[1, 1] = unit_focal_pixel[1]

        # targ_f = self.kitti_uf * np.array((self.width, self.height))
        # re_size = np.round(self.full_res_shape / cts_focal * targ_f).astype(np.intc)
        # re_size = (np.round(re_size / self.downR) * self.downR).astype(np.intc)

        # ck = self.t_folder(folder)
        # if ck not in self.ctsImg_sz_rec.keys():
        #     self.ctsImg_sz_rec[ck] = re_size
        return K
    def get_camK(self, folder, frame_index):
        intrParam, extrParam = self.get_intrin_extrin_param(folder)
        xscale = self.width / self.full_res_shape[0]
        yscale = self.height / self.full_res_shape[1]
        # xscale = 0.5
        # yscale = 0.5
        intrinsic, extrinsic = self.process_InExParam2Matr(intrParam, extrParam, xscale=xscale, yscale=yscale)
        camK = (intrinsic @ extrinsic).astype(np.float32)
        invcamK = np.linalg.inv(camK).astype(np.float32)
        return camK, invcamK, intrinsic.astype(np.float32), extrinsic.astype(np.float32), None
    def get_rescaleFac(self, folder):
        cts_focal, cts_baseline = self.get_cityscape_cam_param(folder)
        kitti_baseline = 0.54
        kitti_unit_focal = 0.58
        cts_unit_focal = cts_focal[0] / self.full_res_shape[0]
        # rescale_fac = (cts_baseline * cts_unit_focal * self.width) / (kitti_baseline * kitti_unit_focal * self.org_width)
        rescale_fac = cts_baseline / kitti_baseline
        return rescale_fac

    def get_seman(self, folder, frame_index, side, do_flip):
        if folder.split('/')[0] == 'train' or folder.split('/')[0] == 'val' or folder.split('/')[0] == 'train_extra':
            seman_path, ins_path = self.get_ins_seman_path(folder)
            color = self.loader(seman_path)
            if do_flip:
                color = color.transpose(pil.FLIP_LEFT_RIGHT)
            seman_label = np.array(color)[:,:,0]
            # ins_label = np.array(self.loader(ins_path))
            # pil.fromarray(((seman_label == 1)*255).astype(np.uint8)).show()
            return seman_label, color
        else:
            return None, None

    # def check_seman(self):
    #     return True

    def process_InExParam2Matr(self, intr, extr, xscale = 1, yscale = 1):
        intrinsic = np.eye(3)
        intrinsic[0,0] = intr['fx']
        intrinsic[1, 1] = intr['fy']
        intrinsic[0, 2] = intr['u0']
        intrinsic[1, 2] = intr['v0']
        intrinsic[2, 2] = 1

        scaleChange = np.eye(3)
        scaleChange[0, 0] = xscale
        scaleChange[1, 1] = yscale
        intrinsic = scaleChange @ intrinsic

        post = np.array([
            [0, -1, 0],
            [0, 0, -1],
            [1, 0, 0]
        ])
        # intrinsic = intrinsic @ post
        intrinsic_ex = np.eye(4)
        intrinsic_ex[0:3,0:3] = intrinsic

        extrinsic = np.eye(4)
        rotM = angle2matrix(pitch = extr['pitch'], roll = extr['roll'], yaw = extr['yaw'])
        trans = np.array([extr['x'], extr['y'], extr['z']])

        rotM = rotM.T
        trans = - rotM @ trans
        extrinsic[0:3,0:3] = rotM
        extrinsic[0:3,3] = trans
        extrinsic[0:3, :] = post @ extrinsic[0:3, :]


        return intrinsic_ex, extrinsic

    def get_cityscape_meta(self, folder):
        self.get_cityscape_meta(folder)

class CITYSCAPERawDataset(CITYSCAPEDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(CITYSCAPERawDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        image_path = os.path.join(self.data_path, self.side_map[side], folder + self.side_map[side] + self.img_ext)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        return None

    def get_cityscape_meta(self, folder):
        image_path = os.path.join(self.data_path, 'disparity', folder + 'disparity' + self.img_ext)
        cts_focal, baseline = self.get_cityscape_cam_param(folder)
        # color = np.array(self.loader(image_path))
        disp = np.array(pil.open(image_path)).astype(np.float)
        mask = np.logical_and(disp!=0, disp>1)
        disp[mask] = (disp[mask] - 1) / 256
        depth = np.zeros(disp.shape)
        depth[mask] = np.clip(cts_focal[0] * baseline / disp[mask], a_min = 0.1, a_max = 100)
        # mask = mask.astype(np.uint8)
        # depth2d = copy.deepcopy(depth)

        # recover 3d points
        intr, extr = self.get_intrin_extrin_param(folder)
        intrinsic, extrinsic = self.process_InExParam2Matr(intr, extr)
        mask_store = copy.deepcopy(mask).astype(np.uint8)
        # mask = mask.flatten()
        # xx, yy = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))
        # xx = xx.flatten()
        # yy = yy.flatten()
        # depthFlat = depth.flatten()
        # oneColumn = np.ones(disp.shape[0] * disp.shape[1])
        # pixelLoc = np.stack([xx[mask] * depthFlat[mask], yy[mask] * depthFlat[mask], depthFlat[mask], oneColumn[mask]], axis=1)
        # cam_coord = (np.linalg.inv(intrinsic) @ pixelLoc.T).T
        # veh_coord = (np.linalg.inv(extrinsic) @ cam_coord.T).T

        #----------- Consider flip------------#
        ctsMeta = dict()
        ctsMeta['depthMap'] = depth
        ctsMeta['mask'] = mask_store
        ctsMeta['intrinsic'] = intrinsic
        ctsMeta['extrinsic'] = extrinsic
        # ctsMeta['veh_coord'] = veh_coord
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(veh_coord[0::100,0], veh_coord[0::100,1], veh_coord[0::100,2], s=0.1)
        # set_axes_equal(ax)
        # pil.fromarray(((depth / depth.max()) * 255).astype(np.uint8)).show()
        return ctsMeta

    def get_cityscape_cam_param(self, folder):
        # camName = os.path.join("camera_trainvaltest", "camera")
        k_path = os.path.join(self.data_path, "camera", folder + "camera.json")
        with open(k_path) as json_file:
            data = json.load(json_file)
            baseline = data['extrinsic']['baseline']
            cts_focal = np.array((data['intrinsic']['fx'], data['intrinsic']['fy'])) # [fx, fy]
        return cts_focal, baseline

    def get_intrin_extrin_param(self, folder):
        # camName = os.path.join("camera_trainvaltest", "camera")
        k_path = os.path.join(self.data_path, "camera", folder + "camera.json")
        with open(k_path) as json_file:
            data = json.load(json_file)
            extrinsic = data['extrinsic']
            intrinsic = data['intrinsic']
        return intrinsic, extrinsic

    def get_ins_seman_path(self, folder):
        if folder.split("/")[0] == 'train' or folder.split("/")[0] == 'val':
            ins_path = os.path.join(self.data_path, "gtFine_processed", folder + "gtFine_instanceTrainIds" + self.img_ext)
            seman_path = os.path.join(self.data_path, "gtFine_processed", folder + "gtFine_labelTrainIds" + self.img_ext)
        elif folder.split("/")[0] == 'train_extra':
            ins_path = os.path.join(self.data_path, "gtCoarse_processed", folder + "gtCoarse_instanceTrainIds" + self.img_ext)
            seman_path = os.path.join(self.data_path, "gtCoarse_processed", folder + "gtCoarse_labelTrainIds" + self.img_ext)
        return seman_path, ins_path