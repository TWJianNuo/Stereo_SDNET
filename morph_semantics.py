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
from auto_morph import *
import Discriminator
import pickle
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
    is_eval_morph = True
    is_cts_bst = True
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
        dataset = datasets.KITTISemanticDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0], 4, is_train=False, tag=opt.dataset)
        train_dataset_predict = datasets.KITTIRAWDataset(
            opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
            [0,'s'], 4, tag='kitti', is_train=False, img_ext='png',
            load_meta=False, is_load_semantics=True,
            is_predicted_semantics=True, load_morphed_depth=False)
        train_dataset_gt = datasets.KITTIRAWDataset(
            opt.data_path, filenames, encoder_dict['height'], encoder_dict['width'],
            [0,'s'], 4, tag='kitti', is_train=False, img_ext='png',
            load_meta=False, is_load_semantics=True,
            is_predicted_semantics=False, load_morphed_depth=False)
    else:
        raise ValueError("No predefined dataset")
    dataloader_predict = DataLoader(train_dataset_predict, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    dataloader_gt = DataLoader(train_dataset_gt, 1, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=False)
    dataloader_predict_iter = iter(dataloader_predict)
    dataloader_gt_iter = iter(dataloader_gt)

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
    depth_pos = '/media/shengjie/other/sceneUnderstanding/bts/result_bts_eigen/raw'


    print("Evaluation starts")

    confMatrix = generateMatrix(args)
    nbPixels = 0
    count255 = 0
    width = 1216
    height = 352
    height_s = int(0.40810811 * height)
    height_e = int(0.99189189 * height)
    width_s = int(0.03594771 * width)
    width_e = int(0.96405229 * width)

    ms = Morph_semantics(height=206, width=1129)
    with torch.no_grad():
        for idx in range(dataloader_gt.__len__()):
            inputs_predict = dataloader_predict_iter.__next__()
            inputs_gt = dataloader_gt_iter.__next__()
            if not is_cts_bst:
                inputs_predict['seman_gt_eval'] = inputs_predict['seman_gt_eval']
            else:
                tcomp = filenames[idx].split(' ')
                path = os.path.join('/media/shengjie/other/sceneUnderstanding/SDNET/cts_best_seman', tcomp[0].split('/')[0] +'_' + tcomp[0].split('/')[1] +  '_' + tcomp[1].zfill(10) + '.png')
                cts_pred = Image.open(path)
                cts_pred = np.array(cts_pred)
                for k in np.unique(cts_pred):
                    cts_pred[cts_pred == k] = labels[k].trainId
                inputs_predict['seman_gt_eval'] = torch.from_numpy(cts_pred).unsqueeze(0)
            # tensor2semantic(inputs_predict['seman_gt_eval'].unsqueeze(1), ind=0).show()
            # tensor2semantic(inputs_gt['seman_gt_eval'].unsqueeze(1), ind=0).show()

            # input_color = inputs[("color", 0, 0)].cuda()
            # outputs = depth_decoder(encoder(input_color),computeSemantic = True, computeDepth = False)
            resized_gt = inputs_gt['seman_gt_eval'].unsqueeze(1)
            # resized_gt = F.interpolate(inputs_gt['seman_gt_eval'].unsqueeze(1).float(), [height, width], mode='nearest')
            # resized_gt = resized_gt.squeeze(1).byte()
            resized_pred = F.interpolate(inputs_predict['seman_gt_eval'].unsqueeze(1).float(), [inputs_gt['seman_gt_eval'].shape[1], inputs_gt['seman_gt_eval'].shape[2]], mode='nearest')
            resized_pred = resized_pred.byte()

            t_height = resized_gt.shape[2]
            t_width = resized_gt.shape[3]
            top_margin = int(t_height - 352)
            left_margin = int((t_width - 1216) / 2)
            resized_gt = resized_gt[:,:,top_margin:top_margin + 352, left_margin:left_margin + 1216]
            resized_pred = resized_pred[:,:,top_margin:top_margin + 352, left_margin:left_margin + 1216]
            # tensor2semantic(resized_gt, ind=0).show()
            # tensor2semantic(resized_pred, ind=0).show()

            resized_rgb = F.interpolate(inputs_gt[('color', 0, 0)], [inputs_gt['seman_gt_eval'].shape[1], inputs_gt['seman_gt_eval'].shape[2]], mode='bilinear', align_corners=True)
            resized_rgb = resized_rgb[:,:,top_margin:top_margin + 352, left_margin:left_margin + 1216]
            pred_depth = get_depth_predict(filenames[idx])
            resized_depth = pred_depth
            # resized_gt = resized_gt.cpu().numpy().astype(np.uint8)
            # resized_pred = resized_pred.cpu().numpy().astype(np.uint8)
            # resized_depth = pred_depth
            # visualize_semantic(gt[0,:,:]).show()
            # visualize_semantic(pred[0,:,:]).show()
            # pred_depth = get_depth_predict(filenames[idx])
            # pred_depth = F.interpolate(pred_depth.float(), [height, width], mode='bilinear', align_corners=True)

            # resized_pred = resized_pred.unsqueeze(1)
            # resized_gt = resized_gt.unsqueeze(1)
            # tensor2semantic(resized_pred, ind=0).show()
            # tensor2semantic(resized_gt, ind=0).show()
            # tensor2disp(1 / pred_depth, vmax=0.15, ind=0).show()
            # disp_map = tensor2disp(1 / pred_depth, vmax=0.15, ind=0)
            # disp_map_combined = combined_2_img(disp_map, tensor2rgb(resized_rgb, ind=0), 0.5)

            pred_depth_cropped = resized_depth[:,:,height_s : height_e, width_s : width_e]
            resized_pred_cropped = resized_pred[:,:,height_s : height_e, width_s : width_e]
            resized_gt_cropped = resized_gt[:,:,height_s : height_e, width_s : width_e]
            resized_rgb_cropped = resized_rgb[:,:,height_s : height_e, width_s : width_e]
            # tensor2semantic(resized_pred_cropped, ind=0).show()
            # tensor2semantic(resized_gt_cropped, ind=0).show()
            # tensor2disp(1 / pred_depth_cropped, vmax=0.15, ind=0).show()
            seman_morphed = ms.morh_semantics(pred_depth_cropped, resized_pred_cropped)
            sv_path = '/media/shengjie/other/sceneUnderstanding/SDNET/visualization/semantic_morph'
            gt_blended = combined_2_img(tensor2semantic(resized_gt_cropped, ind=0), tensor2rgb(resized_rgb_cropped, ind=0), 0.2)
            pred_blended = combined_2_img(tensor2semantic(resized_pred_cropped, ind=0), tensor2rgb(resized_rgb_cropped, ind=0), 0.2)
            morph_blended = combined_2_img(tensor2semantic(seman_morphed, ind=0),
                                          tensor2rgb(resized_rgb_cropped, ind=0), 0.2)
            improved_region = (seman_morphed.cuda().byte() == resized_gt_cropped.cuda().byte()) > (resized_pred_cropped.cuda().byte() == resized_gt_cropped.cuda().byte())
            deterized_region = (seman_morphed.cuda().byte() == resized_gt_cropped.cuda().byte()) < (
                        resized_pred_cropped.cuda().byte() == resized_gt_cropped.cuda().byte())
            improve_blend = combined_2_img(tensor2disp(improved_region, vmax = 1, ind=0),
                                          tensor2rgb(resized_rgb_cropped, ind=0), 0.6)
            deterized_blend = combined_2_img(tensor2disp(deterized_region, vmax = 1, ind=0),
                                          tensor2rgb(resized_rgb_cropped, ind=0), 0.6)
            cat_img = concat_imgs([gt_blended, pred_blended, morph_blended, improve_blend, deterized_blend])
            cat_img.save(os.path.join('/media/shengjie/other/sceneUnderstanding/SDNET/visualization/semantic_morph', str(idx) + '.png'))


            groundTruthNp = resized_gt_cropped.squeeze(1).detach().cpu().numpy()
            if is_eval_morph:
                predictionNp = seman_morphed.byte().squeeze(1).detach().cpu().numpy()
            else:
                predictionNp = resized_pred_cropped.squeeze(1).detach().cpu().numpy()
            nbPixels = nbPixels + groundTruthNp.shape[0] * groundTruthNp.shape[1] * groundTruthNp.shape[2]


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
    if confMatrix.sum() + count255!= nbPixels:
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

class Morph_semantics():
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.tool = grad_computation_tools(batch_size=1, height=self.height,
                                           width=self.width).cuda()
        self.auto_morph = BNMorph(height=self.height, width=self.width, senseRange=20).cuda()
        self.tool.disparityTh = 0.07
    def morh_semantics(self, depth, semantics):
        depth = depth.cuda()
        depth = 1 / depth
        semantics = semantics.cuda()
        foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
        batch_size = semantics.shape[0]
        height = semantics.shape[2]
        width = semantics.shape[3]
        foregroundMapGt = torch.ones([batch_size, 1, height, width],
                                     dtype=torch.uint8, device=torch.device("cuda"))
        for m in foregroundType:
            foregroundMapGt = foregroundMapGt * (semantics != m)
        foregroundMapGt = (1 - foregroundMapGt).float()

        disparity_grad = torch.abs(self.tool.convDispx(depth)) + torch.abs(
            self.tool.convDispy(depth))
        semantics_grad = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(
            self.tool.convDispy(foregroundMapGt))
        disparity_grad = disparity_grad * self.tool.zero_mask
        semantics_grad = semantics_grad * self.tool.zero_mask

        disparity_grad_bin = disparity_grad > self.tool.disparityTh
        semantics_grad_bin = semantics_grad > self.tool.semanticsTh
        # tensor2disp(disparity_grad, ind=0, vmax=0.1).show()
        # tensor2disp(disparity_grad_bin, ind=0, vmax=1).show()
        # fig_seman = tensor2disp(semantics_grad_bin, ind=0, vmax=1)
        morphedx, morphedy, ocoeff = self.auto_morph.find_corresponding_pts(semantics_grad_bin, disparity_grad_bin,
                                                                            pixel_distance_weight=20)
        morphedx = (morphedx / (self.width - 1) - 0.5) * 2
        morphedy = (morphedy / (self.height - 1) - 0.5) * 2
        grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
        seman_morphed = F.grid_sample(semantics.detach().float(), grid, mode = 'nearest', padding_mode="border")
        # tensor2semantic(seman_morphed, ind=0).show()
        # tensor2semantic(semantics, ind=0).show()
        # joint = torch.zeros([height, width, 3])
        # joint[:,:,0] = disparity_grad_bin[0,0,:,:] * 255
        # joint[:, :, 1] = semantics_grad_bin[0, 0, :, :] * 255
        # joint = joint.cpu().numpy().astype(np.uint8)
        # pil.fromarray(joint).show()



        # foregroundMapGt = torch.ones([batch_size, 1, height, width],
        #                              dtype=torch.uint8, device=torch.device("cuda"))
        # for m in foregroundType:
        #     foregroundMapGt = foregroundMapGt * (seman_morphed != m)
        # foregroundMapGt = (1 - foregroundMapGt).float()
        # semantics_grad_morphed = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(
        #     self.tool.convDispy(foregroundMapGt)) * self.tool.zero_mask
        # semantics_grad_bin_morphed = semantics_grad_morphed > self.tool.semanticsTh
        # joint = torch.zeros([height, width, 3])
        # joint[:,:,0] = disparity_grad_bin[0,0,:,:] * 255
        # joint[:, :, 1] = semantics_grad_bin_morphed[0, 0, :, :] * 255
        # joint = joint.cpu().numpy().astype(np.uint8)
        # pil.fromarray(joint).show()
        #
        # selector = ocoeff['orgpts_x'] != -1
        # srcptsx = ocoeff['orgpts_x'][selector]
        # srcptsy = ocoeff['orgpts_y'][selector]
        # dstPtsx = ocoeff['correspts_x'][selector]
        # dstPtsy = ocoeff['correspts_y'][selector]
        # plt.figure()
        # plt.imshow(fig_seman)
        # for i in range(0, srcptsx.shape[0]):
        #     plt.plot([srcptsx[i], dstPtsx[i]], [srcptsy[i], dstPtsy[i]])
        # plt.show()
        return seman_morphed
def get_depth_predict(entry):
    comps = entry.split(' ')
    sub_comp = comps[0].split('/')
    img_path = os.path.join('/media/shengjie/other/sceneUnderstanding/bts/result_bts_eigen/raw', sub_comp[1] + '_' + comps[1].zfill(10) + '.png')
    depth_img = cv2.imread(img_path, -1)
    depth_img_np = np.array(depth_img)
    pred_depth = depth_img_np.astype(np.float32) / 256.0

    pred_depth_ = torch.from_numpy(pred_depth).unsqueeze(0).unsqueeze(0)
    # tensor2disp(1/pred_depth_, vmax=0.15, ind=0).show()
    return pred_depth_
def morph_by_depth(semantics_map, depth_map):
    a = 1


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
