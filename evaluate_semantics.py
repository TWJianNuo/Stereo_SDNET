from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np

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

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

    assert os.path.isdir(opt.load_weights_folder), \
        "Cannot find a folder at {}".format(opt.load_weights_folder)

    print("-> Loading weights from {}".format(opt.load_weights_folder))

    filenames = readlines(os.path.join(splits_dir, opt.split, "val_files.txt"))
    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    if opt.dataset == 'cityscape':
        dataset = datasets.CITYSCAPERawDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, tag=opt.dataset)
    elif opt.dataset == 'kitti':
        dataset = datasets.KITTIRAWDataset(
            opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
            [0,'s'], 4, tag='kitti', is_train=False, img_ext='png',
            load_meta=False, is_load_semantics=True,
            is_predicted_semantics=True, load_morphed_depth=False)
    else:
        raise ValueError("No predefined dataset")
    dataloader = DataLoader(dataset, 16, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
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
    sfx = torch.nn.Softmax(dim=1)

    print("Evaluation starts")

    confMatrix = generateMatrix(args)
    nbPixels = 0
    count255 = 0
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            input_color = inputs[("color", 0, 0)].cuda()
            outputs = depth_decoder(encoder(input_color),computeSemantic = True, computeDepth = False)

            gt = inputs['seman_gt_eval'].cpu().numpy().astype(np.uint8)
            pred = sfx(outputs[('seman', 0)]).detach()
            pred = torch.argmax(pred, dim=1).type(torch.float).unsqueeze(1)
            pred = F.interpolate(pred, [gt.shape[1], gt.shape[2]], mode='nearest')
            pred = pred.squeeze(1).cpu().numpy().astype(np.uint8)
            # visualize_semantic(gt[0,:,:]).show()
            # visualize_semantic(pred[0,:,:]).show()

            groundTruthNp = gt
            predictionNp = pred
            nbPixels = nbPixels + groundTruthNp.shape[0] * groundTruthNp.shape[1] * groundTruthNp.shape[2]

            # encoding_value = max(groundTruthNp.max(), predictionNp.max()).astype(np.int32) + 1
            encoding_value = 256  # precomputed
            encoded = (groundTruthNp.astype(np.int32) * encoding_value) + predictionNp

            values, cnt = np.unique(encoded, return_counts=True)

            for value, c in zip(values, cnt):
                pred_id = value % encoding_value
                gt_id = int((value - pred_id) / encoding_value)
                if pred_id == 255 or gt_id == 255:
                    count255 = count255 + c
                    continue
                if not gt_id in args.evalLabels:
                    printError("Unknown label with id {:}".format(gt_id))
                confMatrix[gt_id][pred_id] += c
            print("Finish %dth batch" % idx)
    if confMatrix.sum() + count255 != nbPixels:
        printError(
            'Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(
                confMatrix.sum(), nbPixels))

    classScoreList = {}
    for label in args.evalLabels:
        labelName = trainId2label[label].name
        classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)
    vals = np.array(list(classScoreList.values()))
    mIOU = np.mean(vals[np.logical_not(np.isnan(vals))])
    # if opt.save_pred_disps:
    #     output_path = os.path.join(
    #         opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
    #     print("-> Saving predicted disparities to ", output_path)
    #     np.save(output_path, pred_disps)

    print("mIOU is %f" % mIOU)



if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
