from __future__ import absolute_import, division, print_function
import os
import argparse
import os
import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
from layers import *
from auto_morph import *
from bnmorph.bnmorph import BNMorph
import pickle
file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Generate kitti eigen predicton options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))
        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti")
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288) # previous is 192
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=960) # previous is 640
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--semantic_minscale",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 choices=[0,1,2,3],
                                 default=[3])
        self.parser.add_argument("--backBone",
                                 nargs="+",
                                 type=str,
                                 help="backbone used",
                                 choices=["unet", "ASPP"],
                                 default=["unet"])
        self.parser.add_argument("--is_sep_train_seman",
                                 help="whether to use different training image for semantic segmentation and deoth",
                                 action="store_true"
                                 )
        self.parser.add_argument("--load_meta",
                                 help="load meta data, for cityscape load gt depth, for kitti load semantics data if have",
                                 action="store_true"
                                 )
        self.parser.add_argument("--val_frequency",
                                 type=int,
                                 default=10,
                                 help="set evaluation frequency"
                                 )
        self.parser.add_argument("--init_discriminator_trainTime",
                                 type=int,
                                 default=5,
                                 help="set evaluation frequency"
                                 )
        self.parser.add_argument("--outputdir",
                                 type=str
                                 )



        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--writeImg",
                                 action="store_true")
        self.parser.add_argument("--debug",
                                 action="store_true")


        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--switchMode",
                                 type=str,
                                 help="turn on switch mode or not",
                                 default="off",
                                 choices=["on", "off"])
        self.parser.add_argument("--semanticCoeff",
                                 type=float,
                                 help="Change semantic loss ratio",
                                 default=1.0,)
        self.parser.add_argument("--isMulChannel",
                                 help="if set, Use multiple depth channel",
                                 action="store_true")
        self.parser.add_argument("--banSemantic",
                                 help="if set, Forbit predict semantic segmentation",
                                 action="store_true")
        self.parser.add_argument("--banDepth",
                                 help="if set, Forbit predict depth",
                                 action="store_true")
        self.parser.add_argument("--selfocclu",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--secondOrderSmooth",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--secondOrderSmoothScale",
                                 help="if set, use self occlusion in training",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--predictboth",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--borderMergeLoss",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--borderMergeLossScale",
                                 type=float,
                                 default=1.0
                                 )
        self.parser.add_argument("--use_kitti_gt_semantics",
                                 help="if set, use kitti gt smeantics",
                                 action="store_true")
        self.parser.add_argument("--borderMorphLoss",
                                 help="if set, use border morphing loss",
                                 action="store_true")
        self.parser.add_argument("--borderMorphScale",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--is_predicted_semantics",
                                 action="store_true")
        self.parser.add_argument("--is_GAN_Training",
                                 action="store_true")
        self.parser.add_argument("--is_toymode",
                                 action="store_true")
        self.parser.add_argument("--borderMorphTime",
                                 type=int,
                                 default=5)
        self.parser.add_argument("--isCudaMorphing",
                                 action="store_true")


        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth"])
        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options

if __name__ == "__main__":
    options = MonodepthOptions()
    opt = options.parse()

    if opt.isCudaMorphing and opt.borderMorphLoss:
        bnmorph = BNMorph(height=opt.height, width=opt.width, sparsityRad=2).cuda()

    outputdir = opt.outputdir
    if not os.path.isdir(outputdir):
        os.mkdir(outputdir)
    splits_dir = os.path.join(os.path.dirname(__file__), "splits")
    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.split, "train_files.txt"))
    # filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                       encoder_dict['height'], encoder_dict['width'],
                                       [0], 4, is_train=False, tag=opt.dataset, img_ext = 'png', load_meta=opt.load_meta, is_load_semantics=opt.use_kitti_gt_semantics, is_predicted_semantics = opt.is_predicted_semantics)

    # dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers,
    #                         pin_memory=True, drop_last=False)

    dataloader = DataLoader(dataset, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, isSwitch=(opt.switchMode == 'on'), isMulChannel=opt.isMulChannel)

    if opt.borderMorphLoss:
        tool = grad_computation_tools(batch_size=opt.batch_size, height=opt.height, width=opt.width).cuda()
        auto_morph = AutoMorph(height=opt.height, width=opt.width)
        foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
        MorphitNum = 5


    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    # encoder.eval()
    depth_decoder.cuda()
    # depth_decoder.eval()
    encoder.train()
    depth_decoder.train()

    pred_disps = []
    mergeDisp = Merge_MultDisp(opt.scales, batchSize = opt.batch_size)

    print("-> Computing predictions with size {}x{}".format(
        encoder_dict['width'], encoder_dict['height']))
    err_tot = 0
    count = 0
    with torch.no_grad():
        for data in dataloader:
            input_color = data[("color", 0, 0)].cuda()

            features = encoder(input_color)
            outputs = dict()
            outputs.update(depth_decoder(features, computeSemantic=True, computeDepth=False))
            outputs.update(depth_decoder(features, computeSemantic=False, computeDepth=True))

            mergeDisp(data, outputs, eval=True)

            # is_skip = True
            # for idx, str_name in enumerate(data['file_add']):
            #     to_sv_path = data['file_add'][idx]
            #     to_sv_comp = to_sv_path.split(' ')
            #
            #     zero_folder = os.path.join(outputdir, to_sv_comp[0].split('/')[0])
            #     first_folder = os.path.join(outputdir, to_sv_comp[0])
            #     fourth_folder_l = os.path.join(outputdir, to_sv_comp[0], 'image_02')
            #     fourth_folder_r = os.path.join(outputdir, to_sv_comp[0], 'image_03')
            #     if to_sv_comp[2] == 'l':
            #         to_sv_path_expanded = os.path.join(fourth_folder_l, to_sv_comp[1].zfill(10) + '.p')
            #     else:
            #         to_sv_path_expanded = os.path.join(fourth_folder_r, to_sv_comp[1].zfill(10) + '.p')
            #
            #     if not os.path.exists(to_sv_path_expanded):
            #         is_skip = False
            #         break
            #
            # if is_skip:
            #     continue

            if opt.borderMorphLoss:
                for key, ipt in data.items():
                    if not (key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta' or key == 'file_add'):
                        data[key] = ipt.to(torch.device("cuda"))

                foregroundMapGt = torch.ones([input_color.shape[0], 1, opt.height, opt.width],
                                             dtype=torch.uint8, device=torch.device("cuda"))
                for m in foregroundType:
                    foregroundMapGt = foregroundMapGt * (data['seman_gt'] != m)
                foregroundMapGt = (1 - foregroundMapGt).float()

                disparity_grad = torch.abs(tool.convDispx(outputs['disp', 0])) + torch.abs(
                    tool.convDispy(outputs['disp', 0]))
                semantics_grad = torch.abs(tool.convDispx(foregroundMapGt)) + torch.abs(
                    tool.convDispy(foregroundMapGt))
                disparity_grad = disparity_grad * tool.zero_mask[0:input_color.shape[0], :, :, :]
                semantics_grad = semantics_grad * tool.zero_mask[0:input_color.shape[0], :, :, :]

                disparity_grad_bin = disparity_grad > tool.disparityTh
                semantics_grad_bin = semantics_grad > tool.semanticsTh

                if opt.isCudaMorphing:
                    morphedx, morphedy, coeff = bnmorph.find_corresponding_pts(disparity_grad_bin, semantics_grad_bin)
                    morphedx = (morphedx / (opt.width - 1) - 0.5) * 2
                    morphedy = (morphedy / (opt.height - 1) - 0.5) * 2
                    grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
                    dispMaps_morphed_torch = F.grid_sample(outputs['disp', 0], grid, padding_mode="border")
                    dispMaps_morphed = list()
                    for i in range(dispMaps_morphed_torch.shape[0]):
                        dispMaps_morphed.append(dispMaps_morphed_torch[i,0,:,:].cpu().numpy())
                else:
                    disparity_grad_bin = disparity_grad_bin.detach().cpu().numpy()
                    semantics_grad_bin = semantics_grad_bin.detach().cpu().numpy()

                    disparityMap_to_processed = outputs['disp', 0].detach().cpu().numpy()
                    dispMaps_morphed = list()
                    changeingRecs = list()
                    for mm in range(input_color.shape[0]):
                        dispMap_morphed, changeingRec = auto_morph.automorph(
                            disparity_grad_bin[mm, 0, :, :], semantics_grad_bin[mm, 0, :, :],
                            disparityMap_to_processed[mm, 0, :, :])
                        dispMaps_morphed.append(dispMap_morphed)
                        changeingRecs.append(changeingRec)

                for idx, morphed_depth_map in enumerate(dispMaps_morphed):
                    # to_save = float2uint8(morphed_depth_map)
                    # to_check = uint82float(to_save)
                    #
                    # err = (np.sum(np.abs(to_check - morphed_depth_map)) / np.sum(morphed_depth_map))
                    # err_tot = err_tot + err
                    count = count + 1


                    # cm = plt.get_cmap('magma')
                    # disretized_show = disretized.astype(np.float) / 255
                    #
                    # slice = (cm(disretized_show) * 255).astype(np.uint8)
                    # slice = slice[:, :, 0:3]
                    # pil.fromarray(slice).show()

                    to_sv_path = data['file_add'][idx]
                    to_sv_comp = to_sv_path.split(' ')

                    zero_folder = os.path.join(outputdir, to_sv_comp[0].split('/')[0])
                    first_folder = os.path.join(outputdir, to_sv_comp[0])
                    fourth_folder_l = os.path.join(outputdir, to_sv_comp[0], 'image_02')
                    fourth_folder_r = os.path.join(outputdir, to_sv_comp[0], 'image_03')
                    if not os.path.isdir(zero_folder):
                        os.mkdir(zero_folder)
                    if not os.path.isdir(first_folder):
                        os.mkdir(first_folder)
                    if not os.path.isdir(fourth_folder_l):
                        os.mkdir(fourth_folder_l)
                    if not os.path.isdir(fourth_folder_r):
                        os.mkdir(fourth_folder_r)

                    if to_sv_comp[2] == 'l':
                        to_sv_path_expanded = os.path.join(fourth_folder_l, to_sv_comp[1].zfill(10) + '.p')
                        to_sv_path_expanded_rgb = os.path.join(fourth_folder_l, to_sv_comp[1].zfill(10) + '.png')
                    else:
                        to_sv_path_expanded = os.path.join(fourth_folder_r, to_sv_comp[1].zfill(10) + '.p')
                        to_sv_path_expanded_rgb = os.path.join(fourth_folder_r, to_sv_comp[1].zfill(10) + '.png')
                    pickle.dump(morphed_depth_map, open(to_sv_path_expanded, "wb"))

                    # pickle.dump(outputs['disp', 0], open(to_sv_path_expanded, "wb"))
                    # color = pickle.load(open(to_sv_path_expanded, "rb")).cuda()
                    # torch.sum(torch.abs(color - outputs['disp', 0]))
                    # to_show = torch.from_numpy(morphed_depth_map).unsqueeze(0).unsqueeze(0)
                    # fig = tensor2disp(to_show, ind=0, vmax=0.08)
                    # fig.save(to_sv_path_expanded_rgb)
                    print("finish : %f" % (count / len(dataset.filenames)))
    # print("tot err %f" % (err_tot / count))
                # dispMaps_morphed = torch.from_numpy(np.stack(dispMaps_morphed, axis=0)).unsqueeze(1).cuda()
                # outputs[("disp", 0)] = dispMaps_morphed
                # tensor2disp(outputs[("disp", 0)], ind=0, vmax=0.08).show()