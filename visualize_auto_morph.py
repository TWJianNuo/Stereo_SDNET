from __future__ import absolute_import, division, print_function
import time
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from layers import *
from utils import readlines
from options import MonodepthOptions
import datasets
import networks
import cityscapesscripts.helpers.labels
from utils import *
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import *
from cityscapesscripts.helpers.labels import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from auto_morph import *

from bnmorph.bnmorph import BNMorph
splits_dir = os.path.join(os.path.dirname(__file__), "splits")
STEREO_SCALE_FACTOR = 5.4


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    viewPythonVer = False
    viewCudaVer = True

    if viewCudaVer:
        bnmorph = BNMorph(height=opt.height, width=opt.width).cuda()

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.split, "val_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    if opt.use_stereo:
        opt.frame_ids.append("s")
    if opt.dataset == 'cityscape':
        dataset = datasets.CITYSCAPERawDataset(opt.data_path, filenames,
                                           opt.height, opt.width, opt.frame_ids, 4, is_train=False, tag=opt.dataset, load_meta=True, direction_left=opt.direction_left)
    elif opt.dataset == 'kitti':
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width, opt.frame_ids, 4, is_train=False, tag=opt.dataset, is_load_semantics=opt.use_kitti_gt_semantics, is_predicted_semantics=opt.is_predicted_semantics, direction_left=opt.direction_left)
    else:
        raise ValueError("No predefined dataset")
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=True)

    encoder = networks.ResnetEncoder(opt.num_layers, False, num_input_images=2)
    if opt.switchMode == 'on':
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, isSwitch=True, isMulChannel=opt.isMulChannel)
    else:
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    viewIndex = 0
    tool = grad_computation_tools(batch_size=opt.batch_size, height=opt.height, width=opt.width).cuda()
    auto_morph = AutoMorph(height=opt.height, width=opt.width)
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            for key, ipt in inputs.items():
                if not(key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta' or key == 'file_add'):
                    inputs[key] = ipt.to(torch.device("cuda"))

            input_color = torch.cat([inputs[("color_aug", 0, 0)], inputs[("color_aug", 's', 0)]], dim=1).cuda()
            # input_color = inputs[("color", 0, 0)].cuda()
            # tensor2rgb(inputs[("color_aug", 0, 0)], ind=0).show()
            # tensor2rgb(inputs[("color_aug", 's', 0)], ind=0).show()
            features = encoder(input_color)
            outputs = dict()
            outputs.update(depth_decoder(features, computeSemantic=True, computeDepth=False))
            outputs.update(depth_decoder(features, computeSemantic=False, computeDepth=True))

            disparityMap = outputs[('mul_disp', 0)]
            depthMap = torch.clamp(disparityMap, max=80)
            fig_seman = tensor2semantic(inputs['seman_gt'], ind=viewIndex, isGt=True)
            fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=viewIndex)
            fig_disp = tensor2disp(disparityMap, ind=viewIndex, vmax=0.1)

            segmentationMapGt = inputs['seman_gt']
            foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17,
                              18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
            foregroundMapGt = torch.ones(disparityMap.shape).cuda().byte()
            for m in foregroundType:
                foregroundMapGt = foregroundMapGt * (segmentationMapGt != m)
            foregroundMapGt = (1 - foregroundMapGt).float()

            disparity_grad = torch.abs(tool.convDispx(disparityMap)) + torch.abs(tool.convDispy(disparityMap))
            semantics_grad = torch.abs(tool.convDispx(foregroundMapGt)) + torch.abs(tool.convDispy(foregroundMapGt))
            disparity_grad = disparity_grad * tool.zero_mask
            semantics_grad = semantics_grad * tool.zero_mask

            disparity_grad_bin = disparity_grad > tool.disparityTh
            semantics_grad_bin = semantics_grad > tool.semanticsTh

            # tensor2disp(disparity_grad_bin, ind=viewIndex, vmax=1).show()
            # tensor2disp(semantics_grad_bin, ind=viewIndex, vmax=1).show()

            if viewPythonVer:
                disparity_grad_bin = disparity_grad_bin.detach().cpu().numpy()
                semantics_grad_bin = semantics_grad_bin.detach().cpu().numpy()

                disparityMap_to_processed = disparityMap.detach().cpu().numpy()[viewIndex, 0, :, :]
                dispMap_morphed, dispMap_morphRec = auto_morph.automorph(disparity_grad_bin[viewIndex,0,:,:], semantics_grad_bin[viewIndex,0,:,:], disparityMap_to_processed)

                fig_disp_processed = visualizeNpDisp(dispMap_morphed, vmax=0.1)
                overlay_processed = pil.fromarray((np.array(fig_disp_processed) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))
                overlay_org = pil.fromarray((np.array(fig_disp) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))
                combined_fig = pil.fromarray(np.concatenate([np.array(overlay_org), np.array(overlay_processed), np.array(fig_disp), np.array(fig_disp_processed)], axis=0))
                combined_fig.save("/media/shengjie/other/sceneUnderstanding/Stereo_SDNET/visualization/border_morph_l2_3/" + str(idx) + ".png")
            if viewCudaVer:
                # morphedx, morphedy = bnmorph.find_corresponding_pts(disparity_grad_bin, semantics_grad_bin, disparityMap, fig_seman, 10)
                # morphedx = (morphedx / (opt.width - 1) - 0.5) * 2
                # morphedy = (morphedy / (opt.height - 1) - 0.5) * 2
                # grid = torch.cat([morphedx, morphedy], dim = 1).permute(0,2,3,1)
                # disparityMap_morphed = F.grid_sample(disparityMap, grid, padding_mode="border")
                # fig_morphed = tensor2disp(disparityMap_morphed, vmax=0.08, ind=0)
                # fig_disp = tensor2disp(disparityMap, vmax=0.08, ind=0)
                # fig_combined = pil.fromarray(np.concatenate([np.array(fig_morphed), np.array(fig_disp)], axis=0))
                # fig_combined.show()
                svpath = os.path.join(opt.load_weights_folder).split('/')
                try:
                    svpath = os.path.join("/media/shengjie/other/sceneUnderstanding/Stereo_SDNET/visualization", svpath[-3])
                    os.mkdir(svpath)
                except FileExistsError:
                    a = 1
                morphedx, morphedy, coeff = bnmorph.find_corresponding_pts(disparity_grad_bin, semantics_grad_bin)
                morphedx = (morphedx / (opt.width - 1) - 0.5) * 2
                morphedy = (morphedy / (opt.height - 1) - 0.5) * 2
                grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
                disparityMap_morphed = F.grid_sample(disparityMap, grid, padding_mode="border")

                fig_morphed = tensor2disp(disparityMap_morphed, vmax=0.08, ind=0)
                fig_disp = tensor2disp(disparityMap, vmax=0.08, ind=0)
                fig_morphed_overlayed = pil.fromarray((np.array(fig_seman) * 0.5 + np.array(fig_morphed) * 0.5).astype(np.uint8))
                fig_disp_overlayed =  pil.fromarray((np.array(fig_seman) * 0.5 + np.array(fig_disp) * 0.5).astype(np.uint8))
                # fig_rgb =  tensor2rgb(inputs[("color", 0, 0)], ind=0)
                # fig_combined = pil.fromarray(np.concatenate([np.array(fig_disp_overlayed), np.array(fig_morphed_overlayed), np.array(fig_disp), np.array(fig_morphed), np.array(fig_rgb)], axis=0))
                fig_combined = pil.fromarray(np.concatenate(
                    [np.array(fig_disp_overlayed), np.array(fig_morphed_overlayed), np.array(fig_disp),
                     np.array(fig_morphed)], axis=0))
                fig_combined.save(
                    os.path.join(svpath,str(idx) + ".png"))
                # fig_combined.save("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/new_morh_alg/" + str(idx) + ".png")
                # tensor2disp(disparityMap_morphed, vmax=0.08, ind=0).show()

if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
