

from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import numpy as np
import cv2
from matplotlib import pyplot as plt


import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import cityscapesscripts.helpers.labels
from utils import *
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import *
from auto_morph import *
import Discriminator
import pickle
from bnmorph.bnmorph import BNMorph
import argparse
from glob import glob
from layers import *
import copy
splits_dir = os.path.join(os.path.dirname(__file__), "splits")

def read_loss(address):
    img_bgr_read = cv2.imread(address)
    B, G, R = np.split(img_bgr_read.astype(np.float), [1, 2], 2)
    arr_read = (B * 65536 + G * 256 + R) / 16777216 * 5
    return arr_read.squeeze()
def save_loss(lossarr, address):
    lossarrnp = lossarr[0,0,:,:].numpy()
    scale_fac = 16777216 / 5 # 2^24
    lossarrnp_scaled = (lossarrnp * scale_fac).astype(np.int64)

    img_bgr = np.zeros((lossarrnp.shape[0], lossarrnp.shape[1], 3), np.int)
    img_bgr[:, :, 0] = lossarrnp_scaled // 65536
    img_bgr[:, :, 1] = (lossarrnp_scaled % 65536) // 256
    img_bgr[:, :, 2] = (lossarrnp_scaled % 65536) % 256
    cv2.imwrite(address, img_bgr)

    # loss_arr_recon = read_loss(address)
    # print(np.mean(np.abs(loss_arr_recon - lossarrnp)))
    # Recon
    # img_bgr = img_bgr.astype(np.float)
    # loss_arr_recon = (img_bgr[:, :, 0] * 65536 + img_bgr[:, :, 1] * 256 + img_bgr[:, :, 2]) / 16777216 * 5
    # np.mean(np.abs(loss_arr_recon - lossarrnp))
def save_img(imgarr, address):
    img_bgr = np.zeros((imgarr.shape[0], imgarr.shape[1], 3), np.int)
    img_bgr[:, :, 0] = imgarr // 256
    img_bgr[:, :, 1] = imgarr % 256
    cv2.imwrite(address, img_bgr)


def read_img(address):
    img_bgr_read = cv2.imread(address)
    B, G, R = np.split(img_bgr_read, [1, 2], 2)
    arr_read = (B * 256 + G).astype(np.uint16).squeeze()
    return arr_read

class Compute_reprojection_loss:
    def __init__(self):
        self.ssim = SSIM()
    def forward(self, pred, target):
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return reprojection_loss

class GenDepthHintsOpt:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data")
        self.parser.add_argument("--save_path",
                                 type=str,
                                 help="path to the training data")
        self.parser.add_argument("--split",
                                 type=str)
        self.parser.add_argument("--split_appendix",
                                 type=str)
        self.parser.add_argument("--window_size",
                                 nargs="+",
                                 type=int,
                                 default=[3, 5, 7, 9])
        self.parser.add_argument("--disparity_range",
                                 nargs="+",
                                 type=int,
                                 default=[48, 96, 192])
        self.parser.add_argument("--width",
                                 type=float,
                                 default=960
                                 )
        self.parser.add_argument("--height",
                                 type=float,
                                 default=288
                                 )
    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

if __name__ == "__main__":
    options = GenDepthHintsOpt()
    opt = options.parse()

    img_dir_names = glob(opt.data_path + '/*/')
    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

    # window_size = [3, 5, 7, 9]
    # disparity_range = [48, 96, 192]
    window_size = opt.window_size
    disparity_range = opt.disparity_range
    right_invalid_val = list()
    left_matcher_list = list()
    right_matcher_list = list()
    xcoord_dicts = dict()
    ycoord_dicts = dict()
    com_reproj_loss = Compute_reprojection_loss()

    tot_time = 0
    tot_num = 0

    if not os.path.exists(opt.save_path):
        os.mkdir(opt.save_path)
    for i in range(len(window_size)):
        for j in range(len(disparity_range)):
            left_matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=disparity_range[j],  # max_disp has to be dividable by 16 f. E. HH 192, 256
                blockSize=window_size[i],
                P1 = int(window_size[i]*window_size[i]*4),
                P2 = int(window_size[i]*window_size[i]*32),
                uniquenessRatio = 10,
                speckleWindowSize = 100,
                speckleRange = 2,
                mode=cv2.StereoSGBM_MODE_HH,
                disp12MaxDiff = 1
            )
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            left_matcher_list.append(left_matcher)
            right_matcher_list.append(right_matcher)
            right_invalid_val.append(-disparity_range[j])


    if opt.split is None:
        for img_dir_name in img_dir_names:
            img_subdir_names = glob(img_dir_name + '*/')
            save_folder_l1 = os.path.join(opt.save_path, img_dir_name.split('/')[-2])
            if not os.path.exists(save_folder_l1):
                os.mkdir(save_folder_l1)
            for img_subdir_name in img_subdir_names:
                imgl_folder_loc = os.path.join(img_subdir_name,'image_02')
                imgr_folder_loc = os.path.join(img_subdir_name,'image_03')

                save_folder_l2 = os.path.join(save_folder_l1, img_subdir_name.split('/')[-2])
                save_folder_l2_l = os.path.join(save_folder_l2,'image_02')
                save_folder_l2_r = os.path.join(save_folder_l2, 'image_03')
                if not os.path.isdir(save_folder_l2):
                    os.mkdir(save_folder_l2)
                if not os.path.isdir(save_folder_l2_l):
                    os.mkdir(save_folder_l2_l)
                if not os.path.isdir(save_folder_l2_r):
                    os.mkdir(save_folder_l2_r)
                # semantic_prediction_folder_loc = os.path.join(semantic_prediction_folder_loc, 'image_' + appendix)
                # semantic_prediction_compose_folder_loc = os.path.join(semantic_prediction_compose_folder_loc,
                #                                                       'image_' + appendix)
                # if not os.path.isdir(semantic_prediction_folder_loc):
                #     os.mkdir(semantic_prediction_folder_loc)
                # if not os.path.isdir(semantic_prediction_compose_folder_loc):
                #     os.mkdir(semantic_prediction_compose_folder_loc)
                # count = 0
                for rgbl_path in glob(os.path.join(imgl_folder_loc, 'data','*.png')):
                    startTime = time.time()
                    rgbr_path = rgbl_path.replace('image_02', 'image_03')
                    imgLs = cv2.imread(rgbl_path, 1)
                    imgRs = cv2.imread(rgbr_path, 1)
                    heightl = imgLs.shape[0]
                    widthl = imgLs.shape[1]
                    heightr = imgRs.shape[0]
                    widthr = imgRs.shape[1]
                    keyword = str(heightl) + str(widthl) + str(heightr) + str(widthr)

                    if keyword not in xcoord_dicts:
                        xx, yy = np.meshgrid(range(widthl), range(heightl), indexing='xy')
                        xcoord_dicts[keyword] = xx.astype(np.float32)
                        ycoord_dicts[keyword] = yy.astype(np.float32)
                    else:
                        xx = xcoord_dicts[keyword]
                        yy = ycoord_dicts[keyword]



                    photo_loss_l = list()
                    photo_loss_r = list()
                    disp_ls = list()
                    disp_rs = list()
                    for k in range(len(left_matcher_list)):
                        imgL = copy.copy(imgLs)
                        imgR = copy.copy(imgRs)
                        displ = left_matcher_list[k].compute(imgL, imgR).astype(np.float32)/16
                        dispr = right_matcher_list[k].compute(imgR, imgL).astype(np.float32)/16

                        selectorl = displ != -1
                        selectorr = dispr != right_invalid_val[k]

                        xx_l = xx - displ * selectorl.astype(np.float32)
                        xx_r = xx - dispr * selectorr.astype(np.float32)

                        xx_l = ((xx_l / (widthl - 1)) - 0.5) * 2
                        xx_r = ((xx_r / (widthr - 1)) - 0.5) * 2

                        yyl = (yy / (heightl - 1) - 0.5) * 2

                        imgL_torch = torch.from_numpy(imgL)[:,:,[2,1,0]].float().unsqueeze(0).permute(0,3,1,2) / 255
                        imgR_torch = torch.from_numpy(imgR)[:,:,[2,1,0]].float().unsqueeze(0).permute(0,3,1,2) / 255
                        grid_l_torch = torch.stack([torch.from_numpy(xx_l).float().unsqueeze(0), torch.from_numpy(yyl).float().unsqueeze(0)], dim=3)
                        imgL_recon =torch.nn.functional.grid_sample(imgR_torch, grid_l_torch, mode='bilinear', padding_mode='reflection')
                        grid_r_torch = torch.stack([torch.from_numpy(xx_r).float().unsqueeze(0), torch.from_numpy(yyl).float().unsqueeze(0)], dim=3)
                        imgR_recon =torch.nn.functional.grid_sample(imgL_torch, grid_r_torch, mode='bilinear', padding_mode='reflection')

                        photolossl = com_reproj_loss.forward(imgL_recon, imgL_torch)
                        photolossr = com_reproj_loss.forward(imgR_recon, imgR_torch)

                        photo_loss_l.append(photolossl)
                        photo_loss_r.append(photolossr)

                        displ[displ == -1] = 0
                        dispr[dispr == right_invalid_val[k]] = 0
                        disp_ls.append(torch.from_numpy(displ).unsqueeze(0).unsqueeze(0))
                        disp_rs.append(torch.from_numpy(dispr).unsqueeze(0).unsqueeze(0))

                        # fig1 = tensor2rgb(imgL_recon, ind=0)
                        # fig2 = tensor2rgb(imgR_recon, ind=0)
                        # fig3 = pil.fromarray(np.concatenate([np.array(fig1), np.array(fig2)], axis=0))
                        # fig3.save("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/stereo_production/" + str(count) + ".png")
                        # tensor2rgb(imgL_recon, ind=0).show()
                        # tensor2rgb(imgL_torch, ind=0).show()
                        # tensor2rgb(imgR_recon, ind=0).show()
                        # tensor2rgb(imgR_torch, ind=0).show()
                        # count = count + 1
                    photo_loss_l = torch.cat(photo_loss_l, dim=0)
                    disp_ls = torch.cat(disp_ls, dim=0)
                    min_loss, min_ind = torch.min(photo_loss_l, dim=0, keepdim=True)
                    predict_l = torch.gather(disp_ls, 0, min_ind)
                    predict_l = predict_l.squeeze(0).squeeze(0).numpy()


                    photo_loss_r = torch.cat(photo_loss_r, dim=0)
                    disp_rs = torch.cat(disp_rs, dim=0)
                    min_val, min_ind = torch.min(photo_loss_r, dim=0, keepdim=True)
                    predict_r = torch.gather(disp_rs, 0, min_ind)
                    predict_r = predict_r.squeeze(0).squeeze(0).numpy()

                    predict_l_int = (predict_l * 16).astype(np.uint16)
                    predict_r_int = (-predict_r * 16).astype(np.uint16)

                    predict_l_sv_path = os.path.join(save_folder_l2_l, rgbl_path.split('/')[-1])
                    save_img(predict_l_int, predict_l_sv_path)

                    predict_r_sv_path = os.path.join(save_folder_l2_r, rgbr_path.split('/')[-1])
                    save_img(predict_r_int, predict_r_sv_path)

                    tot_num = tot_num + 1
                    timeElapsed = time.time() - startTime
                    tot_time = tot_time + timeElapsed
                    print("Finish %d images, average time %f s" %(tot_num, tot_time / tot_num))
                    # plt.figure(0)
                    # plt.imshow(predict_l, 'gray')
                    # plt.show()
                    #
                    # plt.figure(0)
                    # plt.imshow(selectorr, 'gray')
                    # plt.show()

                    # plt.figure(1)
                    # plt.imshow(predict_l, 'gray')
                    # plt.show()
    else:
        split_file_add = os.path.join(os.path.dirname(__file__), 'splits', opt.split, opt.split_appendix + ".txt")
        to_compute_files = readlines(split_file_add)
        for entry_name in to_compute_files:
            date_folder = os.path.join(opt.save_path, entry_name.split(' ')[0].split('/')[0])
            if not os.path.exists(date_folder):
                os.mkdir(date_folder)
            sequence_folder = os.path.join(date_folder, entry_name.split(' ')[0].split('/')[1])
            if not os.path.exists(sequence_folder):
                os.mkdir(sequence_folder)
            left_pred_folder = os.path.join(sequence_folder, "image_02")
            right_pred_folder = os.path.join(sequence_folder, "image_03")
            left_loss_folder = os.path.join(sequence_folder, "image_02_loss")
            right_loss_folder = os.path.join(sequence_folder, "image_03_loss")
            if not os.path.exists(left_pred_folder):
                os.mkdir(left_pred_folder)
            if not os.path.exists(right_pred_folder):
                os.mkdir(right_pred_folder)
            if not os.path.exists(left_loss_folder):
                os.mkdir(left_loss_folder)
            if not os.path.exists(right_loss_folder):
                os.mkdir(right_loss_folder)

            rgbl_path = os.path.join(opt.data_path, entry_name.split(' ')[0], 'image_02', 'data', entry_name.split(' ')[1].zfill(10) + '.png')
            rgbr_path = os.path.join(opt.data_path, entry_name.split(' ')[0], 'image_03', 'data', entry_name.split(' ')[1].zfill(10) + '.png')
            startTime = time.time()
            imgLs = cv2.imread(rgbl_path, 1)
            imgRs = cv2.imread(rgbr_path, 1)
            if (opt.width > 0) and (opt.height > 0):
                imgLs = cv2.resize(imgLs, (int(opt.width), int(opt.height)))
                imgRs = cv2.resize(imgRs, (int(opt.width), int(opt.height)))
            heightl = imgLs.shape[0]
            widthl = imgLs.shape[1]
            heightr = imgRs.shape[0]
            widthr = imgRs.shape[1]
            keyword = str(heightl) + str(widthl) + str(heightr) + str(widthr)

            if keyword not in xcoord_dicts:
                xx, yy = np.meshgrid(range(widthl), range(heightl), indexing='xy')
                xcoord_dicts[keyword] = xx.astype(np.float32)
                ycoord_dicts[keyword] = yy.astype(np.float32)
            else:
                xx = xcoord_dicts[keyword]
                yy = ycoord_dicts[keyword]

            photo_loss_l = list()
            photo_loss_r = list()
            disp_ls = list()
            disp_rs = list()
            for k in range(len(left_matcher_list)):
                imgL = copy.copy(imgLs)
                imgR = copy.copy(imgRs)
                displ = left_matcher_list[k].compute(imgL, imgR).astype(np.float32) / 16
                dispr = right_matcher_list[k].compute(imgR, imgL).astype(np.float32) / 16

                selectorl = displ != -1
                selectorr = dispr != right_invalid_val[k]

                xx_l = xx - displ * selectorl.astype(np.float32)
                xx_r = xx - dispr * selectorr.astype(np.float32)

                xx_l = ((xx_l / (widthl - 1)) - 0.5) * 2
                xx_r = ((xx_r / (widthr - 1)) - 0.5) * 2

                yyl = (yy / (heightl - 1) - 0.5) * 2

                imgL_torch = torch.from_numpy(imgL)[:, :, [2, 1, 0]].float().unsqueeze(0).permute(0, 3, 1,
                                                                                                  2) / 255
                imgR_torch = torch.from_numpy(imgR)[:, :, [2, 1, 0]].float().unsqueeze(0).permute(0, 3, 1,
                                                                                                  2) / 255
                grid_l_torch = torch.stack(
                    [torch.from_numpy(xx_l).float().unsqueeze(0), torch.from_numpy(yyl).float().unsqueeze(0)],
                    dim=3)
                imgL_recon = torch.nn.functional.grid_sample(imgR_torch, grid_l_torch, mode='bilinear',
                                                             padding_mode='reflection')
                grid_r_torch = torch.stack(
                    [torch.from_numpy(xx_r).float().unsqueeze(0), torch.from_numpy(yyl).float().unsqueeze(0)],
                    dim=3)
                imgR_recon = torch.nn.functional.grid_sample(imgL_torch, grid_r_torch, mode='bilinear',
                                                             padding_mode='reflection')

                photolossl = com_reproj_loss.forward(imgL_recon, imgL_torch)
                photolossr = com_reproj_loss.forward(imgR_recon, imgR_torch)

                # tensor2disp(torch.from_numpy(displ).unsqueeze(0).unsqueeze(0), ind=0, percentile=95).show()

                photo_loss_l.append(photolossl)
                photo_loss_r.append(photolossr)

                displ[displ == -1] = 0
                dispr[dispr == right_invalid_val[k]] = 0
                disp_ls.append(torch.from_numpy(displ).unsqueeze(0).unsqueeze(0))
                disp_rs.append(torch.from_numpy(dispr).unsqueeze(0).unsqueeze(0))

                # tensor2rgb(imgL_recon, ind=0).show()
                # tensor2rgb(imgL_torch, ind=0).show()
                # tensor2rgb(imgR_recon, ind=0).show()
                # tensor2rgb(imgR_torch, ind=0).show()

            photo_loss_l = torch.cat(photo_loss_l, dim=0)
            disp_ls = torch.cat(disp_ls, dim=0)
            min_val_left, min_ind = torch.min(photo_loss_l, dim=0, keepdim=True)
            predict_l = torch.gather(disp_ls, 0, min_ind)

            photo_loss_r = torch.cat(photo_loss_r, dim=0)
            disp_rs = torch.cat(disp_rs, dim=0)
            min_val_right, min_ind = torch.min(photo_loss_r, dim=0, keepdim=True)
            predict_r = torch.gather(disp_rs, 0, min_ind)



            # tensor2disp(predict_l, percentile=95, ind=0).show()
            # xx_l_final = xx - predict_l * selectorl.astype(np.float32)
            # xx_r_final = xx - predict_r * selectorr.astype(np.float32)
            # xx_l_final = ((xx_l_final / (widthl - 1)) - 0.5) * 2
            # xx_r_final = ((xx_r_final / (widthr - 1)) - 0.5) * 2
            # grid_l_torch = torch.stack(
            #     [torch.from_numpy(xx_l).float().unsqueeze(0), torch.from_numpy(yyl).float().unsqueeze(0)],
            #     dim=3)
            # imgL_recon = torch.nn.functional.grid_sample(imgR_torch, grid_l_torch, mode='bilinear',
            #                                              padding_mode='reflection')
            # grid_r_torch = torch.stack(
            #     [torch.from_numpy(xx_r).float().unsqueeze(0), torch.from_numpy(yyl).float().unsqueeze(0)],
            #     dim=3)
            # imgR_recon = torch.nn.functional.grid_sample(imgL_torch, grid_r_torch, mode='bilinear',
            #                                              padding_mode='reflection')
            # photolossl = com_reproj_loss.forward(imgL_recon, imgL_torch)
            # photolossr = com_reproj_loss.forward(imgR_recon, imgR_torch)


            predict_l = predict_l.squeeze(0).squeeze(0).numpy()
            predict_r = predict_r.squeeze(0).squeeze(0).numpy()
            predict_l_int = (predict_l * 16).astype(np.uint16)
            predict_r_int = (-predict_r * 16).astype(np.uint16)
            # tensor2disp(predict_l, percentile=95, ind=0).show()
            # tensor2disp(predict_r, percentile=95, ind=0).show()

            predict_l_sv_path = os.path.join(left_pred_folder, rgbl_path.split('/')[-1])
            predict_l_loss_sv_path = os.path.join(left_loss_folder, rgbl_path.split('/')[-1])
            save_img(predict_l_int, predict_l_sv_path)
            save_loss(min_val_left, predict_l_loss_sv_path)

            predict_r_sv_path = os.path.join(right_pred_folder, rgbr_path.split('/')[-1])
            predict_r_loss_sv_path = os.path.join(right_loss_folder, rgbl_path.split('/')[-1])
            save_img(predict_r_int, predict_r_sv_path)
            save_loss(min_val_right, predict_r_loss_sv_path)

            tot_num = tot_num + 1
            timeElapsed = time.time() - startTime
            tot_time = tot_time + timeElapsed
            print("Finish %d images, average time %f s, time left %f hours" % (tot_num, tot_time / tot_num, (len(to_compute_files) - tot_num) * tot_time / tot_num / 60 / 60))