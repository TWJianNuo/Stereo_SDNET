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


def save_loss(lossarrnp, add):
    scale_fac = 16777216
    lossarrnp_scaled = (lossarrnp * scale_fac).astype(np.int64)
    img_bgr = np.zeros((lossarrnp.shape[0], lossarrnp.shape[1], 3), np.int)
    img_bgr[:, :, 0] = lossarrnp_scaled // 65536
    img_bgr[:, :, 1] = (lossarrnp_scaled % 65536) // 256
    img_bgr[:, :, 2] = (lossarrnp_scaled % 65536) % 256
    cv2.imwrite(add, img_bgr)


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
    selfOccluMask = SelfOccluMask().cuda()
    selfOccluMask.th = 0
    if opt.isCudaMorphing and opt.borderMorphLoss:
        bnmorph = BNMorph(height=opt.height, width=opt.width, sparsityRad=2).cuda()
    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:

        opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)

        assert os.path.isdir(opt.load_weights_folder), \
            "Cannot find a folder at {}".format(opt.load_weights_folder)

        filenames = readlines(os.path.join(splits_dir, opt.split_name, opt.appendix_name + ".txt"))
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

        encoder_dict = torch.load(encoder_path)

        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'],
                                           [0, 's'], 4, is_train=False, tag=opt.dataset, img_ext = 'png', load_meta=opt.load_meta, is_load_semantics=opt.use_kitti_gt_semantics, is_predicted_semantics = opt.is_predicted_semantics)

        dataloader = DataLoader(dataset, 2, shuffle=False, num_workers=opt.num_workers, drop_last=False)

        encoder = networks.ResnetEncoder(opt.num_layers, False, num_input_images = 2)
        depth_decoder = networks.DepthDecoder(encoder.num_ch_enc, isSwitch=(opt.switchMode == 'on'), isMulChannel=opt.isMulChannel, outputtwoimage = (opt.outputtwoimage == True))



        model_dict = encoder.state_dict()
        encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
        depth_decoder.load_state_dict(torch.load(decoder_path))

        encoder.cuda()
        encoder.eval()
        depth_decoder.cuda()
        depth_decoder.eval()


        pred_disps = []
        mergeDisp = Merge_MultDisp(opt.scales, batchSize = opt.batch_size)


        count = 0
        tottime = 0

        if not os.path.isdir(opt.output_dir):
            os.mkdir(opt.output_dir)

        with torch.no_grad():
            for data in dataloader:
                # input_colorl = torch.cat([data[("color", 0, 0)], data[("color", 's', 0)]], dim=1).cuda()
                # input_colorr = torch.cat([data[("color", 's', 0)], data[("color", 0, 0)]], dim=1).cuda()
                # input_color = torch.cat([input_colorl, input_colorr], dim=0)
                start = time.time()
                input_color = torch.cat([data[("color", 0, 0)], data[("color", 's', 0)]], dim=1).cuda()
                # tensor2rgb(input_color[:,0:3,:,:], ind=0).show()
                # tensor2rgb(input_color[:, 3:6, :, :], ind=0).show()
                # tensor2rgb(input_color[:, 0:3, :, :], ind=1).show()

                features = encoder(input_color)
                outputs = dict()
                outputs.update(depth_decoder(features, computeSemantic=False, computeDepth=True))

                mergeDisp(data, outputs, eval=True)

                count = count + 1
                scaled_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
                pred_disp = scaled_disp
                pred_disp = pred_disp.cpu()[:, 0].numpy()

                real_scale_disp = scaled_disp * (torch.abs(data[("K", 0)][:, 0, 0] * data["stereo_T"][:, 0, 3]).view(opt.batch_size, 1, 1,1).expand_as(scaled_disp)).cuda()
                SSIMMask = selfOccluMask(real_scale_disp, data["stereo_T"][:, 0, 3].cuda())



                store_path = filenames[data['idx'][0].numpy()].split(' ')
                folder1 = os.path.join(opt.output_dir, store_path[0].split('/')[0])
                folder2 = os.path.join(opt.output_dir, store_path[0])
                folder3 = os.path.join(folder2, 'image_02')
                folder4 = os.path.join(folder2, 'image_03')
                if not os.path.isdir(folder1):
                    os.mkdir(folder1)
                if not os.path.isdir(folder2):
                    os.mkdir(folder2)
                if not os.path.isdir(folder3):
                    os.mkdir(folder3)
                if not os.path.isdir(folder4):
                    os.mkdir(folder4)
                if opt.outputvisualizaiton:
                    folder5 = os.path.join(folder2, 'image_02_compose')
                    folder6 = os.path.join(folder2, 'image_03_compose')
                    if not os.path.isdir(folder5):
                        os.mkdir(folder5)
                    if not os.path.isdir(folder6):
                        os.mkdir(folder6)
                    a = outputs[("disp", 0)] * (1 - SSIMMask)
                    fig1 = tensor2disp(a, ind=0, vmax=0.15)
                    fig2 = tensor2disp(a, ind=1, vmax=0.15)
                    fig1.save(os.path.join(folder5, store_path[1].zfill(10) + '.png'))
                    fig2.save(os.path.join(folder6, store_path[1].zfill(10) + '.png'))
                pathl = os.path.join(folder3, store_path[1].zfill(10) + '.png')
                pathr = os.path.join(folder4, store_path[1].zfill(10) + '.png')

                # fig1 = tensor2disp(outputs[("disp", 0)], ind=1, vmax=0.1)
                # fig2 = tensor2disp(outputs[("disp", 0)] * (1 - SSIMMask), ind=1, vmax=0.1)
                # fig_combined = np.concatenate([np.array(fig1), np.array(fig2)], axis=0)
                # pil.fromarray(fig_combined).show()
                real_scale_disp = real_scale_disp * (1 - SSIMMask)
                stored_disp = real_scale_disp / 960
                save_loss(stored_disp[0, 0, :, :].cpu().numpy(), pathl)
                save_loss(stored_disp[1, 0, :, :].cpu().numpy(), pathr)

                duration = time.time() - start
                tottime = tottime + duration
                print("left time %f hours" % (tottime / count * (len(filenames) - count) / 60 / 60))




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