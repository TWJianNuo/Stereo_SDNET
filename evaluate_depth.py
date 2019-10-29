from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
from glob import glob
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
cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    abs_shift = np.mean(np.abs(gt - pred))

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, abs_shift


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp




def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80
    if opt.isCudaMorphing and opt.borderMorphLoss:
        bnmorph = BNMorph(height=opt.height, width=opt.width, sparsityRad=2).cuda()
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        # print("-> Loading weights from {}".format(opt.load_weights_folder))

        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0, 's'], 4, is_train=False, tag=opt.dataset, img_ext = 'png', load_meta=opt.load_meta, is_load_semantics=opt.use_kitti_gt_semantics, is_predicted_semantics = opt.is_predicted_semantics)

        dataloader = DataLoader(dataset, opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True)

        encoder = networks.ResnetEncoder(opt.num_layers, False, num_input_images = 2)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, isSwitch=(opt.switchMode == 'on'), isMulChannel=opt.isMulChannel, outputtwoimage = (opt.outputtwoimage == True))

        if opt.borderMorphLoss:
            tool = grad_computation_tools(batch_size=opt.batch_size, height=opt.height, width=opt.width).cuda()
            auto_morph = AutoMorph(height=opt.height, width=opt.width)
            foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
            MorphitNum = 5


        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()

        # encoder.train()
        # depth_decoder.train()

        pred_disps = []
        mergeDisp = Merge_MultDisp(opt.scales, batchSize = opt.batch_size)

        # print("-> Computing predictions with size {}x{}".format(
        #     encoder_dict['width'], encoder_dict['height']))
        count = 0
        with torch.no_grad():
            for data in dataloader:
                input_color = torch.cat([data[("color", 0, 0)], data[("color", 's', 0)]], dim=1).cuda()

                if opt.post_process:
                    # Post-processed results require each image to have two forward passes
                    input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                features = encoder(input_color)
                outputs = dict()
                # outputs.update(depth_decoder(features, computeSemantic=True, computeDepth=False))
                outputs.update(depth_decoder(features, computeSemantic=False, computeDepth=True))

                mergeDisp(data, outputs, eval=True)
                # outputs['disp', 0] = F.interpolate(outputs['disp', 0], [opt.height, opt.width], mode='bilinear', align_corners=True)
                # tensor2disp(outputs['disp', 0], ind=0, vmax=0.09).show()
                if opt.borderMorphLoss:
                    for key, ipt in data.items():
                        if not (key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta' or key == 'file_add'):
                            data[key] = ipt.to(torch.device("cuda"))

                    foregroundMapGt = torch.ones([opt.batch_size, 1, opt.height, opt.width],
                                                 dtype=torch.uint8, device=torch.device("cuda"))
                    for m in foregroundType:
                        foregroundMapGt = foregroundMapGt * (data['seman_gt'] != m)
                    foregroundMapGt = (1 - foregroundMapGt).float()

                    disparity_grad = torch.abs(tool.convDispx(outputs['disp', 0])) + torch.abs(
                        tool.convDispy(outputs['disp', 0]))
                    semantics_grad = torch.abs(tool.convDispx(foregroundMapGt)) + torch.abs(
                        tool.convDispy(foregroundMapGt))
                    disparity_grad = disparity_grad * tool.zero_mask
                    semantics_grad = semantics_grad * tool.zero_mask

                    disparity_grad_bin = disparity_grad > tool.disparityTh
                    semantics_grad_bin = semantics_grad > tool.semanticsTh


                    if opt.isCudaMorphing:

                        morphedx, morphedy, coeff = bnmorph.find_corresponding_pts(disparity_grad_bin, semantics_grad_bin)
                        morphedx = (morphedx / (opt.width - 1) - 0.5) * 2
                        morphedy = (morphedy / (opt.height - 1) - 0.5) * 2
                        grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
                        dispMaps_morphed = F.grid_sample(outputs['disp', 0], grid, padding_mode="border")
                    else:
                        disparity_grad_bin = disparity_grad_bin.detach().cpu().numpy()
                        semantics_grad_bin = semantics_grad_bin.detach().cpu().numpy()

                        disparityMap_to_processed = outputs['disp', 0].detach().cpu().numpy()
                        dispMaps_morphed = list()
                        changeingRecs = list()
                        for mm in range(opt.batch_size):
                            dispMap_morphed, changeingRec = auto_morph.automorph(
                                disparity_grad_bin[mm, 0, :, :], semantics_grad_bin[mm, 0, :, :],
                                disparityMap_to_processed[mm, 0, :, :])
                            dispMaps_morphed.append(dispMap_morphed)
                            changeingRecs.append(changeingRec)
                        dispMaps_morphed = torch.from_numpy(np.stack(dispMaps_morphed, axis=0)).unsqueeze(1).cuda()
                    # outputs[("disp", 0)] = dispMaps_morphed[:, 0:1, :, :]
                    # tensor2disp(dispMaps_morphed, ind=0, vmax=0.09).show()

                # print(count)

                # tensor2disp(outputs[("disp", 0)][:, 0:1, :, :], ind=0, vmax=0.1).show()
                count = count + 1
                pred_disp, _ = disp_to_depth(outputs[("disp", 0)][:, 0:1, :, :], opt.min_depth, opt.max_depth)
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                if opt.post_process:
                    N = pred_disp.shape[0] // 2
                    pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle = True)["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "abs_shift"))
    print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")
    if opt.isCudaMorphing and opt.borderMorphLoss:
        bnmorph.print_params()

if __name__ == "__main__":
    options = MonodepthOptions()
    parsed_command = options.parse()
    if parsed_command.load_weights_folders is not None:
        folders_to_eval = glob(parsed_command.load_weights_folders + '*/')
        to_order = list()
        for i in range(len(folders_to_eval)):
            to_order.append(int(folders_to_eval[i].split('/')[-2].split('_')[1]))
        to_order = np.array(to_order)
        to_order_index = np.argsort(to_order)
        for i in to_order_index:
            print(folders_to_eval[i])
            parsed_command.load_weights_folder = folders_to_eval[i]
            evaluate(parsed_command)
    else:
        # alpha_distance_weight = np.arange(0.1, 2, 0.1)
        # pixel_mulline_distance_weight = np.arange(3, 50, 3)
        # alpha_padding = np.arange(0.1, 3, 0.1)
        #
        # for k in alpha_distance_weight:
        #     evaluate(parsed_command, alpha_distance_weight = k)
        # for k in pixel_mulline_distance_weight:
        #     evaluate(parsed_command, pixel_mulline_distance_weight=k)
        # for k in alpha_padding:
        #     evaluate(parsed_command, alpha_padding=k)
        evaluate(parsed_command)