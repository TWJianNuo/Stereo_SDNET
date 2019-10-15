# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from utils import my_Sampler
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import *
from auto_morph import *
import Discriminator
import pickle
from bnmorph.bnmorph import BNMorph
from timeit import default_timer as timer
torch.manual_seed(0)

grads = {}
def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

class Trainer:
    def __init__(self, options):
        self.loss_test = 0
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"
        self.STEREO_SCALE_FACTOR = 5.4

        if self.opt.switchMode == 'on':
            self.switchMode = True
        else:
            self.switchMode = False
        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.semanticCoeff = self.opt.semanticCoeff
        self.sfx = nn.Softmax()
        # self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        # print("1")
        # a = torch.zeros([4, 4, 4]).cuda()
        if not self.opt.use_two_images:
            num_input_images = 1
        else:
            num_input_images = 2
        self.models["encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained", num_input_images = num_input_images)
        # print("1.5")
        # a = torch.zeros([4,4,4]).cuda()
        # print("1.55")
        self.models["encoder"].to(self.device)
        # print("1.6")
        self.parameters_to_train += list(self.models["encoder"].parameters())
        # print("2")
        self.models["depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, isSwitch=self.switchMode, isMulChannel=self.opt.isMulChannel)
        self.models["depth"].to(self.device)

        # print("3")
        self.parameters_to_train += list(self.models["depth"].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        # for param_group in self.model_optimizer.param_groups:
        #     print(param_group['lr'])
        self.morph_optimizer = optim.SGD(self.parameters_to_train, self.opt.learning_rate)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.set_dataset()
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))
        # print("4")
        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.set_layers()
        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("Switch mode on") if self.switchMode else print("Switch mode off")
        print("There are {:d} training items and {:d} validation items\n".format(
            self.train_num, self.val_num))

        # print("5")
        if self.opt.load_weights_folder is not None:
            self.load_model()
        self.save_opts()

        self.sl1 = torch.nn.SmoothL1Loss()
        self.mp2d = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # test
        # self.models["encoder"].eval()
        # self.models["depth"].eval()
        # for batch_idx, inputs in enumerate(self.train_loader):
        #     outputs = dict()
        #     losses = dict()
        #     for key, ipt in inputs.items():
        #         if not (key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta' or key == 'file_add'):
        #             inputs[key] = ipt.to(self.device)
        #     features = self.models["encoder"](inputs["color_aug", 0, 0])
        #     outputs.update(self.models["depth"](features, computeSemantic=False, computeDepth=True))
        #     self.merge_multDisp(inputs, outputs, False)
        #     print(torch.mean(torch.abs(outputs[('mul_disp', 0)] - inputs['depth_morphed'])))


        # For compute disparity bins
        self.disp_range = np.arange(0, 150, 1)
        self.bins = np.zeros(len(self.disp_range) - 1)
    def fork_stable_net_version(self):
        print("forking stable net.....")
        self.models["stable_encoder"] = networks.ResnetEncoder(
            self.opt.num_layers, self.opt.weights_init == "pretrained").cuda()
        self.models["stable_encoder"].load_state_dict(self.models["encoder"].state_dict())
        self.models["stable_depth"] = networks.DepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales, isSwitch=self.switchMode, isMulChannel=self.opt.isMulChannel).cuda()
        self.models["stable_depth"].load_state_dict(self.models["depth"].state_dict())
        self.models["stable_encoder"].train()
        self.models["stable_depth"].train()
        self.set_requires_grad(self.models["stable_encoder"], requires_grad=False)
        self.set_requires_grad(self.models["stable_depth"], requires_grad=False)

        # self.paramrec1 = 0
        # self.paramrec2 = 0
        # paramrec1 = 0
        # paramrec2 = 0
        # for param in self.models["stable_depth"].parameters():
        #     paramrec1 = paramrec1 + torch.sum(torch.abs(param.data))
        # for param in self.models["stable_encoder"].parameters():
        #     paramrec2 = paramrec2 + torch.sum(torch.abs(param.data))
        #
        # for param in self.models["stable_depth"].parameters():
        #     self.paramrec1 = self.paramrec1 + torch.sum(torch.abs(param.data))
        # for param in self.models["stable_encoder"].parameters():
        #     self.paramrec2 = self.paramrec2 + torch.sum(torch.abs(param.data))
        print("Finished")

    def set_layers(self):
        """properly handle layer initialization under multiple dataset situation
        """
        train_stage_span = self.opt.normalTrainingTime
        self.train_mode_span = np.array([train_stage_span, 0])
        self.semanticLoss = Compute_SemanticLoss(min_scale = self.opt.semantic_minscale[0])
        self.merge_multDisp = Merge_MultDisp(self.opt.scales, batchSize = self.opt.batch_size, isMulChannel = self.opt.isMulChannel)
        self.compsurfnorm = {}
        self.backproject_depth = {}
        self.project_3d = {}
        if self.opt.selfocclu:
            self.selfOccluMask = SelfOccluMask().cuda()
        tags = list()
        for t in self.format:
            tags.append(t[0])
        for p, tag in enumerate(tags):
            height = self.format[p][1]
            width = self.format[p][2]
            for n, scale in enumerate(self.opt.scales):
                h = height // (2 ** scale)
                w = width // (2 ** scale)

                self.backproject_depth[(tag, scale)] = BackprojectDepth(self.opt.batch_size, h, w)
                self.backproject_depth[(tag, scale)].to(self.device)

                self.project_3d[(tag, scale)] = Project3D(self.opt.batch_size, h, w)
                self.project_3d[(tag, scale)].to(self.device)

        if self.opt.secondOrderSmooth:
            self.compSecOrder = SecondOrderGrad().cuda()

        if self.opt.borderMergeLoss:
            self.foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18] # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
            self.borderMerge = expBinaryMap(self.format[0][1], self.format[0][2], self.opt.batch_size).cuda()

        if self.opt.borderMorphLoss:
            self.topk_kval = int(self.opt.width * self.opt.height * 0.995) # trained on 0.99
            self.tool = grad_computation_tools(batch_size=self.opt.batch_size, height=self.opt.height, width=self.opt.width).cuda()
            if self.opt.more_sv:
                self.isSaved = 0
            if not self.opt.isCudaMorphing:
                self.auto_morph = AutoMorph(height=self.opt.height, width=self.opt.width)
            else:
                self.auto_morph = BNMorph(height=self.opt.height, width=self.opt.width, senseRange = 20).cuda()
            self.foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
            # self.MorphitNum = 5
            self.train_mode_span[1] = self.opt.borderMorphTime
            if self.opt.gaussian_sigma >= 0.01:
                gaussWeights = get_gaussian_kernel_weights(kernel_size=7, sigma=self.opt.gaussian_sigma)
                self.gaussConv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=7, padding=3, bias=False)
                self.gaussConv.weight = nn.Parameter(gaussWeights.repeat(1, 1, 1, 1), requires_grad=False)
                self.gaussConv = self.gaussConv.cuda()
            else:
                self.gaussConv = None
            self.totLoss = 0
            # if self.opt.is_GAN_Training:
            #     self.netD = Discriminator.PixelDiscriminator(input_nc = 2, ndf=64, norm_layer=nn.BatchNorm2d).cuda()
            #     self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
            #     self.criterionGAN = networks.GANLoss('vanilla').to(self.device)
            #     self.criterionL1 = torch.nn.MSELoss()
        if self.opt.concrete_reg:
            in_ex_set = pickle.load(open("in_ex_set.p", "rb"))
            self.csp = ComputeSurroundingPixDistance(height=self.opt.height, width=self.opt.width,
                                                batch_size=self.opt.batch_size, intrinsic_set=in_ex_set['intrinsic_sets'],
                                                extrinsic_set=in_ex_set['extrinsic_sets'])

        if self.opt.read_stereo:
            self.compute_subpixel_metric = Subpixel_metric().cuda()
    def set_dataset(self):
        """properly handle multiple dataset situation
        """
        datasets_dict = {
            "kitti": datasets.KITTIRAWDataset,
            "kitti_odom": datasets.KITTIOdomDataset,
            "cityscape": datasets.CITYSCAPERawDataset,
            "joint": datasets.JointDataset
                         }
        dataset_set = self.opt.dataset.split('+')
        split_set = self.opt.split.split('+')
        datapath_set = self.opt.data_path.split('+')
        assert len(dataset_set) == len(split_set), "dataset and split should have same number"
        stacked_train_datasets = list()
        stacked_val_datasets = list()
        train_sample_num = np.zeros(len(dataset_set), dtype=np.int)
        val_sample_num = np.zeros(len(dataset_set), dtype=np.int)
        for i, d in enumerate(dataset_set):
            initFunc = datasets_dict[d]
            fpath = os.path.join(os.path.dirname(__file__), "splits", split_set[i], "{}_files.txt")
            train_filenames = readlines(fpath.format("train"))
            val_filenames = readlines(fpath.format("val"))
            img_ext = '.png' if self.opt.png else '.jpg'
            is_load_semantics = True if d == 'cityscape' or (d == 'kitti' and self.opt.use_kitti_gt_semantics) else False

            train_dataset = initFunc(
                datapath_set[i], train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, tag=dataset_set[i], is_train = (not self.opt.is_toymode), img_ext=img_ext, load_meta=self.opt.load_meta, is_load_semantics=is_load_semantics, is_predicted_semantics=self.opt.is_predicted_semantics, load_morphed_depth=self.opt.load_morphed_depth, read_stereo=self.opt.read_stereo, direction_left=self.opt.direction_left)
            train_sample_num[i] = train_dataset.__len__()
            stacked_train_datasets.append(train_dataset)

            val_dataset = initFunc(
                datapath_set[i], val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, tag=dataset_set[i], is_train=False, img_ext=img_ext, load_meta=self.opt.load_meta, is_load_semantics=is_load_semantics, is_predicted_semantics=self.opt.is_predicted_semantics, direction_left=self.opt.direction_left)
            val_sample_num[i] = val_dataset.__len__()
            stacked_val_datasets.append(val_dataset)

        initFunc = datasets_dict['joint']
        self.joint_dataset_train = initFunc(stacked_train_datasets)
        joint_dataset_val = initFunc(stacked_val_datasets)

        if not self.opt.group_two:
            self.trainSample = my_Sampler(train_sample_num, self.opt.batch_size) # train sampler is used for multi-stage training
            valSample = my_Sampler(val_sample_num, self.opt.batch_size)
        else:
            self.trainSample = my_Sampler_group2(train_sample_num, self.opt.batch_size) # train sampler is used for multi-stage training
            valSample = my_Sampler(val_sample_num, self.opt.batch_size)
        if not self.opt.no_train_shuffle:
            self.train_loader = DataLoader(
                self.joint_dataset_train, self.opt.batch_size, shuffle=False, sampler=self.trainSample,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        else:
            self.train_loader = DataLoader(
                self.joint_dataset_train, self.opt.batch_size, shuffle=False,
                num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            joint_dataset_val, self.opt.batch_size, shuffle=False, sampler=valSample,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        num_train_samples = self.joint_dataset_train.__len__()
        self.train_num = self.joint_dataset_train.__len__()
        self.val_num = joint_dataset_val.__len__()
        self.format = self.joint_dataset_train.format
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs


        mu_std = pickle.load(open("mu_var", "rb"))
        self.mu = torch.from_numpy(mu_std[:,:,0]).unsqueeze(0).unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1).cuda()
        self.std = torch.from_numpy(mu_std[:, :, 1]).unsqueeze(0).unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1).cuda()
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()
            # self.set_dataset()
            # f = plt.figure()
            # _ = plt.bar(np.arange(len(self.bins)), self.bins / np.sum(self.bins))
            # plt.xlabel("disparity in pixels")
            # plt.ylabel("percentage")
            # plt.title("Disparity statistics from 3000 training samples")
            # plt.savefig("disparity_hist.png", dpi = 400)


    def supervised_with_morph(self, inputs):
        # outputs = dict()
        # losses = dict()
        # for key, ipt in inputs.items():
        #     if not (key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta' or key == 'file_add'):
        #         inputs[key] = ipt.to(self.device)
        # features = self.models["encoder"](inputs["color_aug", 0, 0])
        # outputs.update(self.models["depth"](features, computeSemantic=False, computeDepth=True))
        # self.merge_multDisp(inputs, outputs, False)


        if not self.opt.inline_finetune:
            outputs = dict()
            losses = dict()
            for key, ipt in inputs.items():
                if not (key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta' or key == 'file_add'):
                    inputs[key] = ipt.to(self.device)
            features = self.models["encoder"](inputs["color_aug", 0, 0])
            outputs.update(self.models["depth"](features, computeSemantic=False, computeDepth=True))
            self.merge_multDisp(inputs, outputs)

            if self.gaussConv is not None:
                blurred_predict = self.gaussConv(outputs['disp', 0])
                blurred_gt = self.gaussConv(inputs['depth_morphed'])
            else:
                blurred_predict = outputs['disp', 0]
                blurred_gt = inputs['depth_morphed']
            diffMap = (blurred_predict - blurred_gt)**2
            losses['totLoss'] = torch.mean(diffMap) * 1e3
            losses["similarity_loss"] = losses["totLoss"]
            # print(self.totLoss / self.step)
            self.morph_optimizer.zero_grad()
            losses['totLoss'].backward()
            self.morph_optimizer.step()

        else:
            # Test
            outputs, losses = self.process_batch(inputs)

            # self.csp.get_3d_pts(depthmap=outputs[('depth', 0, 0)], intrinsic=inputs['realIn'], extrinsic=inputs['realEx'], semantic=inputs['seman_gt'])
            # tensor2rgb(inputs['color', 0, 0], ind=0).show()
            # tensor2rgb(inputs['color', 0, 0], ind=1).show()
            # tensor2rgb(inputs['color', 0, 0], ind=2).show()
            # tensor2rgb(inputs['color', 0, 0], ind=3).show()
            # tensor2rgb(inputs['color', 0, 0], ind=4).show()
            # tensor2rgb(inputs['color', 0, 0], ind=5).show()
            # with torch.no_grad():
            #     stable_disp = self.models['stable_depth'](self.models['stable_encoder'](inputs[('color', 0, 0)]))[('mul_disp', 0)]
            stable_disp = outputs['disp', 0].detach()
            foregroundMapGt = torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width],
                                         dtype=torch.uint8, device=torch.device("cuda"))
            for m in self.foregroundType:
                foregroundMapGt = foregroundMapGt * (inputs['seman_gt'] != m)
            foregroundMapGt = (1 - foregroundMapGt).float()

            disparity_grad = torch.abs(self.tool.convDispx(outputs['disp', 0])) + torch.abs(
                self.tool.convDispy(outputs['disp', 0]))
            semantics_grad = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(
                self.tool.convDispy(foregroundMapGt))
            disparity_grad = disparity_grad * self.tool.zero_mask
            semantics_grad = semantics_grad * self.tool.zero_mask

            disparity_grad_bin = disparity_grad > self.tool.disparityTh
            semantics_grad_bin = semantics_grad > self.tool.semanticsTh

            # morphedx, morphedy, ocoeff = self.auto_morph.find_corresponding_pts(disparity_grad_bin, semantics_grad_bin)

            morphedx, morphedy, ocoeff = self.auto_morph.find_corresponding_pts(disparity_grad_bin, semantics_grad_bin, pixel_distance_weight = 20)

            morphedx = (morphedx / (self.opt.width - 1) - 0.5) * 2
            morphedy = (morphedy / (self.opt.height - 1) - 0.5) * 2
            grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
            # dispMaps_morphed = F.grid_sample(outputs['disp', 0].detach(), grid, padding_mode="border")
            dispMaps_morphed = F.grid_sample(stable_disp.detach(), grid, padding_mode="border")
            outputs['dispMaps_morphed'] = dispMaps_morphed
            ssim_morph = self.compute_reprojection_loss(dispMaps_morphed, outputs['disp', 0])

            if not self.opt.use_ssim_compare_mask:
                kth_val, kth_ind = torch.kthvalue(ssim_morph.cpu().view(self.opt.batch_size, 1, -1), dim=2, k=self.topk_kval)
                kth_val = kth_val.cuda()
                selector_mask = (ssim_morph > kth_val.view(-1, 1, 1, 1).expand(-1,1,self.opt.height, self.opt.width)).float()
                losses["similarity_loss"] = torch.sum(ssim_morph * selector_mask * outputs['grad_proj_msak'] * (1 - outputs['ssimMask'])) / (torch.sum(selector_mask) + 1)

                # a = self.ssim(dispMaps_morphed, outputs['disp', 0])
                # losses["similarity_loss"] = torch.mean(torch.abs(dispMaps_morphed - outputs['disp', 0]) * outputs['grad_proj_msak'] * (1-outputs['ssimMask']))
                # losses["similarity_loss"] = torch.mean(ssim_morph * outputs['grad_proj_msak'] * (1 - outputs['ssimMask']))
                losses['totLoss'] = losses["similarity_loss"] * self.opt.l1_weight + losses['totLoss']
                # tensor2disp(outputs['grad_proj_msak'] * (1-outputs['ssimMask']), vmax=1, ind=0).show()
            else:
                with torch.no_grad():
                    ssim_val_predict = self.compute_reprojection_loss(outputs[('color', 's', 0)], inputs[('color', 0, 0)])

                    scaledDisp, depth = disp_to_depth(dispMaps_morphed, self.opt.min_depth, self.opt.max_depth)

                    frame_id = "s"
                    T = inputs["stereo_T"]
                    cam_points = self.backproject_depth[('kitti', 0)](
                        depth, inputs[("inv_K", 0)])
                    pix_coords = self.project_3d[('kitti', 0)](
                        cam_points, inputs[("K", 0)], T)
                    morphed_rgb = F.grid_sample(inputs[("color", frame_id, 0)], pix_coords, padding_mode="border")
                    ssim_val_morph = self.compute_reprojection_loss(morphed_rgb, inputs[('color', 0, 0)])

                    # pred_nonorm = self.project_3d[('kitti', 0)].forward_no_normalize(cam_points, inputs[("K", 0)], T)
                    # gt = self.backproject_depth[('kitti', 0)].id_coords[0, :, :].unsqueeze(0).expand(
                    #     self.opt.batch_size, -1, -1, -1)
                    # diffs = torch.abs(pred_nonorm[:,:,:, 0].unsqueeze(1) - gt)
                    # self.bins += np.histogram(diffs.flatten().detach().cpu().numpy(), bins=self.disp_range)[0]

                    # for t in range(self.opt.batch_size):
                    #     gt = self.backproject_depth[('kitti', 0)].id_coords[0,:,:].unsqueeze(0).expand(self.opt.batch_size,-1,-1,-1)
                    #     pred = pred_nonorm[t, :, :, 0]
                    #     diffs = torch.abs(pred - gt)
                    #     tensor2disp(diffs.view(1,1,288,960), percentile=95, ind=0).show()
                    #     print("Max diff is %f, min diff is %f" % (diffs.max(), diffs.min()))
                    # tensor2rgb(morphed_rgb, ind=0).show()
                    # tensor2rgb(outputs[('color', 's', 0)], ind=0).show()
                    # tensor2disp(((ssim_val_predict - 1.01 * ssim_val_morph) > 0) + (outputs['ssimMask'] > 0), ind=0, vmax=1).show()
                    # dilated_ssimMask = self.mp2d(outputs['ssimMask'])
                    # selector_mask = ((ssim_val_predict - ssim_val_morph) > 0).float() * outputs['grad_proj_msak'] * (
                    #             1 - dilated_ssimMask)
                    # kth_val, kth_ind = torch.kthvalue(ssim_morph.cpu().view(self.opt.batch_size, 1, -1), dim=2,
                    #                                   k=int(self.opt.width * self.opt.height * 0.97))
                    # kth_val = kth_val.cuda()
                    # selector_mask_select = (ssim_morph > kth_val.view(-1, 1, 1, 1).expand(-1, 1, self.opt.height,
                    #                                                                self.opt.width)).float()
                    # selector_mask = selector_mask * selector_mask_select
                    # outputs['selector_mask'] = selector_mask
                    # selector_mask = (ssim_val_predict - ssim_val_morph > 0).float() * outputs['grad_proj_msak'] * (1 - outputs['ssimMask'])
                    selector_mask = (ssim_val_predict - 1.05 * ssim_val_morph > 0).float() * outputs['grad_proj_msak']
                    # ssim_morph = ssim_morph * selector_mask
                    # tensor2disp(ssim_morph, ind=0, percentile=99.5).show()
                    outputs['selector_mask'] = selector_mask

                losses["similarity_loss"] = torch.sum(torch.log(1 + torch.abs(dispMaps_morphed - outputs['disp', 0])) * selector_mask) / (torch.sum(selector_mask) + 1)
                # losses['totLoss'] = losses["similarity_loss"] * self.opt.l1_weight + losses['totLoss']
                if not self.opt.delay_open:
                    losses['totLoss'] = losses["similarity_loss"] * self.opt.l1_weight + losses['totLoss']
                else:
                    if self.epoch > 2:
                        losses['totLoss'] = losses["similarity_loss"] * self.opt.l1_weight + losses['totLoss']


                if self.opt.read_stereo:
                    with torch.no_grad():
                        disparity_sgm = inputs['img_disparity']
                        min_disp = 1 / self.opt.max_depth
                        max_disp = 1 / self.opt.min_depth
                        activation_sgm = (disparity_sgm / (0.1 * 0.58 * self.opt.width) - min_disp) / (max_disp - min_disp)
                        activation_sgm = torch.clamp(activation_sgm, min=0, max=1)


                        org_pixel_loc = self.backproject_depth[('kitti', 0)].id_coords
                        moved_x = org_pixel_loc[0,:,:].unsqueeze(0).expand(self.opt.batch_size, -1, -1) + torch.sign(inputs['stereo_T'][:, 0, 3]).view(self.opt.batch_size,1,1).expand(-1, self.opt.height, self.opt.width) * disparity_sgm.squeeze(1)
                        moved_y = org_pixel_loc[1, :, :].unsqueeze(0).expand(self.opt.batch_size, -1, -1)
                        moved_x = ((moved_x / (self.opt.width - 1)) - 0.5) * 2
                        moved_y = ((moved_y / (self.opt.height - 1)) - 0.5) * 2
                        grid_samples = torch.stack([moved_x, moved_y], dim=3)
                        img_recon = torch.nn.functional.grid_sample(inputs[('color_aug', 's', 0)], grid_samples, mode='bilinear',padding_mode='reflection')
                        depthhints_ssim = self.compute_reprojection_loss(img_recon, inputs[('color_aug', 0, 0)])

                        # tensor2disp(activation_sgm, vmax=0.09, ind=0).show()
                        # tensor2disp(outputs[('disp', 0)], vmax=0.09, ind=0).show()
                        # tensor2rgb(img_recon, ind=0).show()

                    depthhints_loss = 0
                    for pp in self.opt.scales:
                        to_compute_mask = (depthhints_ssim < outputs[('to_optimize', pp)]).float() * (1 - outputs['ssimMask']) * inputs['img_disparity_valmask']
                        # tensor2disp(inputs['img_disparity_valmask'], vmax=1, ind=0).show()
                        # tensor2disp((1 - outputs['ssimMask']), vmax=1, ind=0).show()
                        # tensor2disp(to_compute_mask, vmax=1, ind=0).show()
                        depthhints_loss = depthhints_loss + torch.sum(torch.abs(activation_sgm - outputs[('disp', pp)]) * to_compute_mask) / (torch.sum(to_compute_mask) + 1)
                    depthhints_loss = depthhints_loss / len(self.opt.scales)
                    losses['totLoss'] = depthhints_loss * 1 + losses['totLoss']

                    # setereo_ssim = self.compute_subpixel_metric.forward(img_recon, inputs[('color_aug', 0, 0)])
                    # compare_ssim = self.compute_subpixel_metric.forward(outputs[('color', 's', 0)], inputs[('color_aug', 0, 0)])

                    # tensor2disp(setereo_ssim, percentile=95, ind=0).show()
                    # tensor2disp(compare_ssim, percentile=95, ind=0).show()
                    # setereo_ssim = self.compute_reprojection_loss(img_recon, inputs[('color_aug', 0, 0)])
                    # compare_ssim = self.compute_reprojection_loss(img_recon, inputs[('color_aug', 0, 0)])

                    # img_metric1 = self.compute_subpixel_metric.forward(img_recon, inputs[('color_aug', 0, 0)])
                    # img_metric1 = self.compute_subpixel_metric.forward(outputs[('color', 's', 0)], inputs[('color_aug', 0, 0)])

                    # self.sum_kernel = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(21, 3),padding=(10, 1), bias=False)
                    # sum_kernel_weights = torch.ones([1,1,21,3])
                    # self.sum_kernel.weight = nn.Parameter(sum_kernel_weights, requires_grad=False)
                    # self.sum_kernel = self.sum_kernel.cuda()
                    #
                    # setereo_ssim = self.sum_kernel(setereo_ssim)
                    # compare_ssim = self.sum_kernel(compare_ssim)
                    #
                    # tensor2disp(setereo_ssim, percentile=95, ind=0).show()
                    # tensor2disp(compare_ssim, percentile=95, ind=0).show()

                    # self.arbssim = ArbSSIM(kehight = 21, kwidth = 5)
                    # setereo_ssim_arb = self.compute_arb_reprojection_loss(img_recon, inputs[('color_aug', 0, 0)])
                    # compare_ssim_arb = self.compute_arb_reprojection_loss(outputs[('color', 's', 0)], inputs[('color_aug', 0, 0)])

                    # stereo_selector = (setereo_ssim >  2 * compare_ssim).float() * inputs['img_disparity_valmask'] * (1-outputs['ssimMask'])
                    # stereo_selector_view = stereo_selector.cpu().numpy()[0,0,:,:]
                    # fig1 = tensor2disp(stereo_selector, ind=0, vmax=0)
                    # fig2 = tensor2rgb(outputs[('color', 's', 0)], ind=0)
                    # fig2_arr = np.array(fig2)
                    # fig2_arr[:,:,0][stereo_selector_view == 1] = 255
                    # fig2_arr[:, :, 1][stereo_selector_view == 1] = 0
                    # fig2_arr[:, :, 2][stereo_selector_view == 1] = 0
                    # fig2 = pil.fromarray(fig2_arr)
                    # fig2.show()
                    # tensor2rgb(img_recon, ind=0).show()
                    # tensor2rgb(outputs[('color', 's', 0)], ind=0).show()
                    # tensor2rgb(inputs[('color_aug', 0, 0)], ind=0).show()

                    # losses['totLoss'] = losses['totLoss'] + torch.sum(stereo_selector * torch.log(1 + torch.abs(activation_sgm - outputs['disp', 0])))/ (torch.sum(stereo_selector) + 1)
                    # losses['totLoss'] = losses['totLoss'] + tmploss / len(self.opt.scales) * 0.1


                    # tensor2disp(activation_sgm, ind=0, vmax=0.1).show()
                    # tensor2disp(outputs['dispMaps_morphed'], ind=0, vmax=0.1).show()
                    # tensor2disp(inputs['img_ssimloss'], ind=0, percentile=95).show()
                    # tensor2rgb(img_recon, ind=0).show()
                    # tensor2rgb(inputs[('color', 0, 0)], ind=0).show()
                    # tensor2rgb(outputs[('color', 's', 0)], ind=0).show()
                    # a = 1
                    # fig3 = tensor2disp(activation_sgm, ind=0, vmax=0.1)
                    # fig4 = tensor2disp(outputs[('disp', 0)], ind=0, vmax=0.1)
                    #
                    # fig5 = tensor2rgb(inputs[('color', 0, 0)], ind=0)
                    # fig6 = tensor2rgb(outputs[('color', 's', 0)], ind=0)
                    # fig7 = tensor2rgb(img_recon, ind=0)
                    #
                    # fig_combined = pil.fromarray(np.concatenate([np.array(fig5), np.array(fig6), np.array(fig7)], axis=0))
                    # fig_combined2 = pil.fromarray(np.concatenate([np.array(fig2), np.array(fig3), np.array(fig4)], axis=0))
                    # figt = pil.fromarray(np.concatenate([np.array(fig_combined), np.array(fig_combined2)], axis=1))

                self.model_optimizer.zero_grad()
                losses['totLoss'].backward()
                self.model_optimizer.step()


                # Test
                # a = 1
                # dispmap_test = outputs['disp', 0].clone()
                # dispmap_test[inputs['seman_gt'] == 10] = 1e-5
                # tensor2disp(dispmap_test, ind=0, vmax=0.08).show()
                #
                # scaledDisp, depth = disp_to_depth(dispmap_test, self.opt.min_depth, self.opt.max_depth)
                # frame_id = "s"
                # T = inputs["stereo_T"]
                # cam_points = self.backproject_depth[('kitti', 0)](
                #     depth, inputs[("inv_K", 0)])
                # pix_coords = self.project_3d[('kitti', 0)](
                #     cam_points, inputs[("K", 0)], T)
                # test_rgb = F.grid_sample(inputs[("color", frame_id, 0)], pix_coords, padding_mode="border")
                #
                # ssim1 = self.compute_reprojection_loss(outputs[('color', 's', 0)], inputs[('color', 0, 0)])
                # ssim2 = self.compute_reprojection_loss(test_rgb, inputs[('color', 0, 0)])
                # diff = (ssim2 - ssim1)
                # diff = diff.clamp(min=0)
                # tensor2disp(diff, vmax=0.03, ind=0).show()
                #
                # visualizeGrad(ssim1 - ssim2, percentile=95, viewind=0).show()
                # torch.mean(ssim1[inputs['seman_gt'] == 10])
                # torch.mean(ssim2[inputs['seman_gt'] == 10])
                # tensor2disp(ssim1 > ssim2, vmax=1, ind=0).show()
                # tensor2disp(inputs['seman_gt'] == 10, vmax=1, ind=0).show()
                # tensor2rgb(inputs[('color', 0, 0)], ind=0).show()
                # tensor2rgb(inputs[('color', 's', 0)], ind=0).show()


                    # fig1 = tensor2disp(outputs['dispMaps_morphed'], vmax=0.08, ind=0)
                    # fig2 = tensor2disp(outputs['disp', 0], vmax=0.08, ind=0)
                # rowsearchSpan = np.arange(0, self.opt.width)
                # colsearchSpan = np.arange(0, self.opt.height)
                # xx, yy = np.meshgrid(rowsearchSpan, colsearchSpan)
                # xx = torch.from_numpy(xx).unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1).cuda().float()
                # yy = torch.from_numpy(yy).unsqueeze(0).expand(self.opt.batch_size, -1, -1, -1).cuda().float()
                # diff_in_x = (xx - (morphedx/2 + 0.5)*(self.opt.width - 1))
                # fig5 = visualizeGrad(diff_in_x, percentile=95, viewind=0)
                # plt.figure()
                # plt.imshow(fig5)
                # selector = ocoeff['orgpts_x'][0,0,:,:] > -1
                # org_xxx = ocoeff['orgpts_x'][0,0,:,:][selector].detach().cpu().numpy()
                # org_yyy = ocoeff['orgpts_y'][0, 0, :, :][selector].detach().cpu().numpy()
                # dst_xxx = ocoeff['correspts_x'][0, 0, :, :][selector].detach().cpu().numpy()
                # dst_yyy = ocoeff['correspts_y'][0, 0, :, :][selector].detach().cpu().numpy()
                # for i in range(org_xxx.shape[0]):
                #     plt.plot([org_xxx[i], dst_xxx[i]], [org_yyy[i], dst_yyy[i]])
                #
                #
                # fig1 = tensor2disp(outputs['dispMaps_morphed'], vmax=0.08, ind=0)
                # fig2 = tensor2disp(outputs['disp', 0], vmax=0.08, ind=0)
                # fig3 = visualizeGrad(outputs['dispMaps_morphed'] - outputs['disp', 0], percentile=95, viewind=0)
                # fig4 = Image.open('/media/shengjie/other/sceneUnderstanding/SDNET/visualization/morph_ssiml1_3/0.png')
                # fig_combined = pil.fromarray(np.concatenate([np.array(fig4), np.array(fig3), np.array(fig5)], axis=0))
                # fig_combined_path = pil.fromarray(np.array(fig_combined)[288 * 2 : 288 * 3, :, :])
                #
                # ratio = np.clip(np.array(fig3).min(axis=2, keepdims=True) / 255 + 0.3, 0, 1)
                # fig1_weighted = pil.fromarray((np.array(fig_combined_path) * np.concatenate([ratio, ratio, ratio], axis=2)).astype(np.uint8))
                # plt.figure()
                # plt.imshow(fig1_weighted)
                # tensor2disp(torch.from_numpy(ratio).unsqueeze(0).unsqueeze(0), vmax=1, ind=0).show()

                ###############################################################################################3

                # with torch.no_grad():
                #     stable_disp = self.models['stable_depth'](self.models['stable_encoder'](inputs[('color', 0, 0)]))[('mul_disp', 0)]
                # fig1 = visualizeGrad(outputs['dispMaps_morphed'] - outputs['disp', 0], percentile=95, viewind=0)
                # fig2 = tensor2disp(stable_disp, vmax=0.08, ind=0)
                # fig_combined = pil.fromarray((np.array(fig1) * 0.5 + np.array(fig2) * 0.5).astype(np.uint8))
                # self.fork_stable_net_version()
                # self.opt.load_weights_folder = '/media/shengjie/other/sceneUnderstanding/SDNET/tmp/ssim_takeTurnNoFork_lr_3/models/weights_1'


                # vind = 0
                # pertile = 95
                # self.morph_optimizer.zero_grad()
                # self.model_optimizer.zero_grad()
                # outputs['disp', 0].register_hook(save_grad('disp'))
                # losses['totLoss'].backward()
                # fig1 = visualizeGrad(-grads['disp'], percentile=pertile, viewind=vind)
                # plt.figure()
                # plt.imshow(fig1)
                #
                # fig2 = tensor2disp(outputs['disp', 0], ind=vind, vmax=0.08)
                # fig3 = pil.fromarray((np.array(fig2) * 0.7 + np.array(fig1) * 0.3).astype(np.uint8))

            # cm = plt.get_cmap('bwr')
            # vind = 0
            # pertile = 95
            # self.morph_optimizer.zero_grad()
            # self.model_optimizer.zero_grad()
            # outputs, losses = self.process_batch(inputs)
            # outputs['disp', 0].register_hook(save_grad('disp'))
            #
            # ssim_morph = self.compute_reprojection_loss(dispMaps_morphed, outputs['disp', 0])
            # kth_val, kth_ind = torch.kthvalue(ssim_morph.cpu().view(self.opt.batch_size, 1, -1), dim=2, k=self.topk_kval)
            # kth_val = kth_val.cuda()
            # selector_mask = (
            #             ssim_morph > kth_val.view(-1, 1, 1, 1).expand(-1, 1, self.opt.height, self.opt.width)).float()
            # (torch.sum(ssim_morph * selector_mask * outputs['grad_proj_msak'] * (1 - outputs['ssimMask'])) / (torch.sum(selector_mask) + 1) * self.opt.l1_weight + losses['totLoss']).backward()
            # visualizeGrad(-grads['disp'], percentile=pertile, viewind=vind).save('1.png')
            # visualizeGrad(dispMaps_morphed - outputs['disp', 0], percentile=pertile, viewind=vind).save('2.png')
            # tensor2disp(selector_mask, ind=vind, vmax=1).save('3.png')
            #
            # predict_bck = outputs['disp', 0].clone()
            # morph_bck = dispMaps_morphed.clone()
            #
            # outputs[('disp', 0)] = predict_bck
            # self.generate_images_pred(inputs, outputs)
            # ssim_loss_predict = self.compute_reprojection_loss(outputs[('color', 's', 0)], inputs[('color', 0, 0)])
            # outputs[('disp', 0)] = morph_bck
            # self.generate_images_pred(inputs, outputs)
            # ssim_loss_morph = self.compute_reprojection_loss(outputs[('color', 's', 0)], inputs[('color', 0, 0)])
            # tensor2disp(ssim_loss_predict, ind=vind, percentile=95).show()
            # tensor2disp(ssim_loss_morph, ind=vind, percentile=95).show()
            # visualizeGrad(ssim_loss_predict - ssim_loss_morph, percentile=pertile, viewind=vind).show()
            # new_selector = (ssim_loss_predict - ssim_loss_morph) *( ((ssim_loss_predict - ssim_loss_morph) > 0).float())
            # tensor2disp(new_selector, ind=vind, vmax= 0.1).show()
            # torch.sum((ssim_loss_predict - ssim_loss_morph) * selector_mask)
            #
            # self.model_optimizer.step()
            # last_round_re_bk = outputs['disp', 0].clone()
            # outputs, losses = self.process_batch(inputs)
            # visualizeGrad(outputs['disp', 0] - last_round_re_bk, percentile=pertile, viewind=vind).show()





            """
            # To visualize the gradient
            diff = (dispMaps_morphed - outputs['disp', 0]) ** 2
            grad_mask = diff > 0.0005
            cm = plt.get_cmap('bwr')
            vind = 0
            pertile = 95



            last_round_re_bk = torch.zeros_like(outputs[('mul_disp', 0)])
            for i in range(100):
                self.morph_optimizer.zero_grad()
                self.model_optimizer.zero_grad()
                outputs, losses = self.process_batch(inputs)

                disparity_grad = torch.abs(self.tool.convDispx(outputs['disp', 0])) + torch.abs(
                    self.tool.convDispy(outputs['disp', 0]))
                semantics_grad = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(
                    self.tool.convDispy(foregroundMapGt))
                disparity_grad = disparity_grad * self.tool.zero_mask
                semantics_grad = semantics_grad * self.tool.zero_mask

                disparity_grad_bin = disparity_grad > self.tool.disparityTh
                semantics_grad_bin = semantics_grad > self.tool.semanticsTh
                morphedx, morphedy, x = self.auto_morph.find_corresponding_pts(disparity_grad_bin, semantics_grad_bin)
                morphedx = (morphedx / (self.opt.width - 1) - 0.5) * 2
                morphedy = (morphedy / (self.opt.height - 1) - 0.5) * 2
                grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)

                dispMaps_morphed = F.grid_sample(outputs['disp', 0].detach(), grid, padding_mode="border")
                outputs['disp', 0].register_hook(save_grad('disp'))
                ((torch.mean(torch.abs(dispMaps_morphed - outputs['disp', 0])) * 2  + losses['totLoss'])).backward()
                # losses['totLoss'].backward()
                fig_grad = visualizeGrad(-grads['disp'], percentile=pertile, viewind=vind)
                self.model_optimizer.step()
                fig_contrast = visualizeGrad(outputs['disp', 0] - last_round_re_bk, percentile=pertile, viewind=vind)
                last_round_re_bk = outputs['disp', 0].clone()


                viewIndex = 0
                fig_seman = tensor2semantic(inputs['seman_gt'], ind=viewIndex, isGt=True)
                fig_disp = tensor2disp(outputs['disp', 0], ind=viewIndex, vmax=0.09)
                overlay_org = pil.fromarray((np.array(fig_disp) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))


                fig_disp_morphed = tensor2disp(dispMaps_morphed, ind=viewIndex, vmax=0.09)
                overlay_dst = pil.fromarray(
                    (np.array(fig_disp_morphed) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))
                combined_fig = pil.fromarray(np.concatenate(
                    [np.array(overlay_org), np.array(fig_disp), np.array(overlay_dst), np.array(fig_disp_morphed), np.array(fig_contrast), np.array(fig_grad)],
                    axis=0))
                combined_fig.save(
                    "/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderMerge/" + str(
                        i) + ".png")
                print(i)



            # visualize the gradient of photometric loss
            self.morph_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            outputs, losses = self.process_batch(inputs)
            outputs['disp', 0].register_hook(save_grad('disp'))
            losses['totLoss'].backward()
            visualizeGrad(grads['disp'], percentile=pertile, viewind=vind).show()


            self.morph_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            outputs, losses = self.process_batch(inputs)
            # outputs['disp', 0] = outputs['disp', 0] * 0
            dispMaps_morphed = F.grid_sample(outputs['disp', 0].detach(), grid, padding_mode="border")
            outputs['disp', 0].register_hook(save_grad('disp'))
            torch.mean((outputs['disp', 0] - dispMaps_morphed.detach()) ** 2 * 1e3).backward()
            fig_grad_reg = visualizeGrad(grads['disp'], percentile=pertile, viewind=vind)
            fig_overlay_disp = pil.fromarray((np.array(tensor2disp(outputs['disp', 0], vmax=0.08, ind=0)) * 0.8 + np.array(fig_grad_reg) * 0.2).astype(np.uint8))


            cm = plt.get_cmap('bwr')
            vind = 0
            pertile = 95
            self.morph_optimizer.zero_grad()
            self.model_optimizer.zero_grad()
            outputs, losses = self.process_batch(inputs)
            dispMaps_morphed = F.grid_sample(outputs['disp', 0].detach(), grid, padding_mode="border")
            outputs['disp', 0].register_hook(save_grad('disp'))
            # normed_morph = (dispMaps_morphed - self.mu) / self.std
            # normed_pred = (outputs['disp', 0] - self.mu) / self.std
            # (self.sl1(normed_pred, normed_morph) * 0.00001).backward()
            # ((self.sl1(normed_pred, normed_morph) + losses['totLoss']) * 0.1).backward()
            # (torch.mean((dispMaps_morphed - outputs['disp', 0]) ** 2 * 1e3) + losses['totLoss']).backward()
            # ((torch.mean(torch.abs(dispMaps_morphed - outputs['disp', 0])) * 20  + losses['totLoss'])).backward()
            (torch.mean(self.compute_reprojection_loss(dispMaps_morphed, outputs['disp', 0])) * 4 + losses['totLoss']).backward()
            # torch.mean(torch.abs(dispMaps_morphed - outputs['disp', 0])).backward()
            # losses['totLoss'].backward()
            # (torch.mean((dispMaps_morphed - outputs['disp', 0]) ** 2 * 1e3)).backward()
            visualizeGrad(-grads['disp'], percentile=pertile, viewind=vind).show()
            visualizeGrad(dispMaps_morphed - outputs['disp', 0], percentile=pertile, viewind=vind).show()
            # tensor2disp(torch.abs(grads['disp']), percentile=99.9, ind=0).show()
            self.model_optimizer.step()
            last_round_re_bk = outputs['disp', 0].clone()
            outputs, losses = self.process_batch(inputs)
            visualizeGrad(outputs['disp', 0] - last_round_re_bk, percentile=pertile, viewind=vind).show()
            


            """
            # disp_grad_numpy = grads['disp'].cpu().numpy()[vind, 0, :, :]
            #
            # selector_pos = disp_grad_numpy > 0
            # pos_bar = np.percentile(disp_grad_numpy[selector_pos], pertile)
            # disp_grad_numpy[selector_pos] = disp_grad_numpy[selector_pos] / pos_bar / 2
            #
            # selector_neg = disp_grad_numpy < 0
            # neg_bar = -np.percentile(-disp_grad_numpy[selector_neg], pertile)
            # disp_grad_numpy[selector_neg] = -disp_grad_numpy[selector_neg] / neg_bar / 2
            #
            # disp_grad_numpy = disp_grad_numpy + 0.5
            # colorMap = cm(disp_grad_numpy)
            # pil.fromarray((colorMap * 255).astype(np.uint8)).show()





            # tensor2disp(inputs['depth_morphed'], vmax=0.1, ind=1).show()
            # tensor2disp(dispMaps_morphed, vmax=0.1, ind=1).show()
            # tensor2disp(outputs['disp', 0], vmax=0.1, ind=1).show()
            # tensor2disp(disparity_grad_bin, vmax=0.1, ind=1).show()
            # tensor2disp(semantics_grad_bin, vmax=0.1, ind=1).show()
            # tensor2disp((outputs['disp', 0] - dispMaps_morphed)**2, vmax=0.0003, ind=1).show()
            # losses['totLoss'] = losses['totLoss'] + torch.mean(((outputs['disp', 0] - dispMaps_morphed) / torch.mean(outputs['disp', 0]))**2) * 1e3
            # outputs['dispMaps_morphed'] = dispMaps_morphed
            # with torch.no_grad():
            #     foregroundMapGt = torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width],
            #                                  dtype=torch.uint8, device=torch.device("cuda"))
            #     for m in self.foregroundType:
            #         foregroundMapGt = foregroundMapGt * (inputs['seman_gt'] != m)
            #     foregroundMapGt = (1 - foregroundMapGt).float()
            #
            #     supervise_output = dict()
            #     supervise_output.update(self.models["stable_depth"](self.models["stable_encoder"](inputs['color_aug', 0, 0][0:1,:,:,:])))
            #
            #     disparity_grad = torch.abs(self.tool.convDispx(supervise_output['mul_disp', 0])) + torch.abs(
            #         self.tool.convDispy(supervise_output['mul_disp', 0]))
            #     semantics_grad = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(
            #         self.tool.convDispy(foregroundMapGt))
            #     disparity_grad = disparity_grad * self.tool.zero_mask
            #     semantics_grad = semantics_grad * self.tool.zero_mask
            #
            #     disparity_grad_bin = disparity_grad > self.tool.disparityTh
            #     semantics_grad_bin = semantics_grad > self.tool.semanticsTh
            #     if not self.opt.isCudaMorphing:
            #         disparity_grad_bin = disparity_grad_bin.detach().cpu().numpy()
            #         semantics_grad_bin = semantics_grad_bin.detach().cpu().numpy()
            #
            #         disparityMap_to_processed = supervise_output['mul_disp', 0].detach().cpu().numpy()
            #         dispMaps_morphed = list()
            #         for mm in range(self.opt.batch_size):
            #             dispMap_morphed, changeingRec = self.auto_morph.automorph(
            #                 disparity_grad_bin[mm, 0, :, :], semantics_grad_bin[mm, 0, :, :],
            #                 disparityMap_to_processed[mm, 0, :, :])
            #             dispMaps_morphed.append(dispMap_morphed)
            #         dispMaps_morphed = torch.from_numpy(np.stack(dispMaps_morphed, axis=0)).unsqueeze(1).cuda()
            #     else:
            #         morphedx, morphedy = self.auto_morph.find_corresponding_pts(disparity_grad_bin[0:1,:,:,:], semantics_grad_bin[0:1,:,:,:])
            #         morphedx = (morphedx / (self.opt.width - 1) - 0.5) * 2
            #         morphedy = (morphedy / (self.opt.height - 1) - 0.5) * 2
            #         grid = torch.cat([morphedx, morphedy], dim=1).permute(0, 2, 3, 1)
            #         dispMaps_morphed = F.grid_sample(supervise_output['mul_disp', 0], grid, padding_mode="border").detach()
            #
            #     supervise_output['dispMaps_morphed'] = dispMaps_morphed
            #     outputs['dispMaps_morphed'] = dispMaps_morphed
            #
            # if self.gaussConv is not None:
            #     blurred_predict = self.gaussConv(outputs['disp', 0])
            #     blurred_gt = self.gaussConv(supervise_output['dispMaps_morphed'])
            # else:
            #     blurred_predict = outputs['disp', 0]
            #     blurred_gt = supervise_output['dispMaps_morphed']
            # diffMap = (blurred_predict[0:1,:,:,:] - blurred_gt) ** 2
            # losses['totLoss'] = torch.mean(diffMap) * 1e3
            # losses["similarity_loss"] = losses["totLoss"]

            # check
            # dispMaps_morphed_test = torch.zeros_like(dispMaps_morphed)
            # for k in range(self.opt.batch_size):
            #     morphedx_test, morphedy_test = self.auto_morph.find_corresponding_pts(disparity_grad_bin[k:k+1,:,:,:], semantics_grad_bin[k:k+1,:,:,:])
            #     morphedx_test = (morphedx_test / (self.opt.width - 1) - 0.5) * 2
            #     morphedy_test = (morphedy_test / (self.opt.height - 1) - 0.5) * 2
            #     grid_test = torch.cat([morphedx_test, morphedy_test], dim=1).permute(0, 2, 3, 1)
            #     dispMaps_morphed_test[k:k+1, :, :, :] = F.grid_sample(outputs['disp', 0][k:k+1,:,:,:], grid_test, padding_mode="border")
            # print(torch.mean(torch.abs(outputs['disp', 0] - outputs['disp', 0])))
            # tensor2disp(dispMaps_morphed, vmax=0.1, ind=1).show()
            # tensor2disp(dispMaps_morphed_test, vmax=0.1, ind=1).show()
            # tensor2disp(outputs['disp', 0], vmax=0.1, ind=1).show()

            # check
            # print(torch.mean(torch.abs(outputs[('disp', 0)] - supervise_output['mul_disp', 0])))
            # print(torch.mean(torch.abs(outputs[('mul_disp', 0)] - inputs['depth_morphed'])))
            # print(torch.mean(torch.abs(supervise_output[('mul_disp', 0)] - inputs['depth_morphed'])))
            # tensor2disp(inputs['depth_morphed'], vmax=0.08, ind=0).show()
            # tensor2disp(outputs[('disp', 0)], vmax=0.08, ind=0).show()
            # tensor2disp(torch.abs(inputs['depth_morphed'] - outputs[('disp', 0)]), vmax=0.008, ind=0).show()
            # print(torch.mean(inputs[('color', 0, 0)])) # 0.329

            # print("loss is %f" % losses['totLoss'])
            #
            # blurred_predict = self.gaussConv(outputs['disp', 0])
            # blurred_gt = self.gaussConv(inputs['depth_morphed'])
            # diffMap = (blurred_predict - blurred_gt)**2
            # losses['totLoss'] = torch.mean(diffMap) * 1e3
            # losses["similarity_loss"] = losses["totLoss"]
            # print("loss2 is %f" % losses['totLoss'])
            #
            # diffComp1 = torch.mean(torch.abs(supervise_output['dispMaps_morphed'][0:1,:,:,:] - inputs['depth_morphed'][0:1,:,:,:]))
            # tensor2disp(supervise_output['dispMaps_morphed'], ind=0, vmax=0.08).show()
            # tensor2disp(inputs['depth_morphed'], ind=0, vmax=0.08).show()
            # print("Diff morph is %f" % diffComp1)

            # loss1 = torch.mean(diffMap) * 1e3
            # blurred_predict = self.gaussConv(outputs['disp', 0])
            # blurred_gt = self.gaussConv(inputs['depth_morphed'])
            # diffMap = (blurred_predict - blurred_gt)**2
            # loss2 = torch.mean(diffMap) * 1e3
            # print("loss diff %f, loss is %f" % (torch.abs(loss1 - loss2) / loss1, loss1))
            # torch.mean(torch.abs(outputs['dispMaps_morphed'] - inputs['depth_morphed']))

            # diffMap = self.compute_reprojection_loss(blurred_predict, blurred_gt)
            # losses['totLoss'] = torch.mean(diffMap) * 1e0
            # losses["similarity_loss"] = losses["totLoss"]
            # tensor2disp(diffMap, ind=0, vmax=0.22).show()


        # tensor2disp(blurred_predict, ind=0, vmax=0.08).show()
        # tensor2disp(blurred_gt, ind=0, vmax=0.08).show()
        # tensor2disp(inputs['depth_morphed'], ind=0, vmax=0.08).show()
        # gaussWeights = get_gaussian_kernel_weights(kernel_size=3, sigma=1)

        # if not self.opt.is_GAN_Training:
        #     blurred_predict = self.gaussConv(outputs['disp', 0])
        #     blurred_gt = self.gaussConv(inputs['depth_morphed'])
        #
        #     tensor2disp(blurred_predict, ind=0, vmax=0.08).show()
        #     tensor2disp(blurred_gt, ind=0, vmax=0.08).show()
        #     tensor2disp(inputs['depth_morphed'], ind=0, vmax=0.08).show()
        #
        #     diffMap = (blurred_predict - blurred_gt)**2
        #
        #     losses['totLoss'] = torch.mean(diffMap) * 1e3
        #     losses["similarity_loss"] = losses["totLoss"]
        # else:
        #     self.set_requires_grad(self.models["encoder"], False)
        #     self.set_requires_grad(self.models["depth"], False)
        #     self.set_requires_grad(self.netD, True)
        #
        #     self.optimizer_D.zero_grad()
        #     fake_AB = torch.cat((outputs['disp', 0], foregroundMapGt), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        #     pred_fake = self.netD(fake_AB.detach())
        #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
        #
        #     real_AB = torch.cat((inputs['depth_morphed'], foregroundMapGt), 1)
        #     pred_real = self.netD(real_AB.detach())
        #     self.loss_D_real = self.criterionGAN(pred_real, True)
        #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        #     self.loss_D.backward()
        #     self.optimizer_D.step()
        #
        #     self.set_requires_grad(self.models["encoder"], True)
        #     self.set_requires_grad(self.models["depth"], True)
        #     self.set_requires_grad(self.netD, False)
        #
        #     pred_fake = self.netD(fake_AB)
        #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        #     self.loss_G_L1 = self.criterionL1(outputs['disp', 0], inputs['depth_morphed']) * 100
        #     self.loss_G = self.loss_G_L1
        #     losses['totLoss'] = self.loss_G * 1e1
        #     losses["similarity_loss"] = torch.mean((outputs['disp', 0] - inputs['depth_morphed'])**2) * 1e3
        #     print(losses["similarity_loss"])
        # self.loss_test = self.loss_test + losses['totLoss']
        # print("mean loss is %f" % (self.loss_test / self.step))
        # self.morph_optimizer.zero_grad()

        return outputs, losses

    def supervised_with_photometric(self, inputs):
        outputs, losses = self.process_batch(inputs)
        self.model_optimizer.zero_grad()
        losses["totLoss"].backward()
        self.model_optimizer.step()
        return outputs, losses
    def decide_training_mode(self):
        # current_stage = np.mod(self.epoch, np.sum(self.train_mode_span))
        self.fork_stable_net_version()
        return False
        # if self.epoch > 100:
        #     return False
        # else:
        #     return True
        # if not self.opt.is_photometric_finetune:
        #     return False
        # else:
        #     if np.mod(self.epoch, 2) == 0:
        #         return False
        #     else:
        #         return True
        # if current_stage < self.train_mode_span[0]:
        #     return True
        # else:
        #     if current_stage == self.train_mode_span[0]:
        #         self.fork_stable_net_version()
        #     return False

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        # is_normal_training = self.decide_training_mode()
        is_normal_training = True
        self.model_lr_scheduler.step()

        if is_normal_training:
            print("Using normal training")
        else:
            print("Using morph training")

        self.set_train()
        # adjust by changing the sampler
        for batch_idx, inputs in enumerate(self.train_loader):


            if self.opt.more_sv:
                if int(self.step / 5000) > self.isSaved :
                    self.save_model(int(self.step / 5000))
                    self.isSaved = int(self.step / 5000)

            before_op_time = time.time()



            # if self.opt.borderMorphLoss and self.opt.is_GAN_Training:
            #     for key, ipt in inputs.items():
            #         if not (key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta'):
            #             inputs[key] = ipt.to(self.device)
            #
            #     for k in range(self.MorphitNum):
            #         features = self.models["encoder"](inputs["color_aug", 0, 0])
            #         suboutputs = dict()
            #         suboutputs.update(self.models["depth"](features, computeSemantic=False, computeDepth=True))
            #         self.merge_multDisp(inputs, suboutputs)
            #
            #         foregroundMapGt = torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width],
            #                                      dtype=torch.uint8, device=torch.device("cuda"))
            #         for m in self.foregroundType:
            #             foregroundMapGt = foregroundMapGt * (inputs['seman_gt'] != m)
            #         foregroundMapGt = (1 - foregroundMapGt).float()
            #
            #         disparity_grad = torch.abs(self.tool.convDispx(suboutputs['disp', 0])) + torch.abs(
            #             self.tool.convDispy(suboutputs['disp', 0]))
            #         semantics_grad = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(
            #             self.tool.convDispy(foregroundMapGt))
            #         disparity_grad = disparity_grad * self.tool.zero_mask
            #         semantics_grad = semantics_grad * self.tool.zero_mask
            #
            #         disparity_grad_bin = disparity_grad > self.tool.disparityTh
            #         semantics_grad_bin = semantics_grad > self.tool.semanticsTh
            #
            #
            #         disparity_grad_bin = disparity_grad_bin.detach().cpu().numpy()
            #         semantics_grad_bin = semantics_grad_bin.detach().cpu().numpy()
            #
            #         disparityMap_to_processed = suboutputs['disp', 0].detach().cpu().numpy()
            #         dispMaps_morphed = list()
            #         changeingRecs = list()
            #         for mm in range(self.opt.batch_size):
            #             dispMap_morphed, changeingRec = self.auto_morph.automorph(
            #                 disparity_grad_bin[mm, 0, :, :], semantics_grad_bin[mm, 0, :, :],
            #                 disparityMap_to_processed[mm, 0, :, :])
            #             dispMaps_morphed.append(dispMap_morphed)
            #             changeingRecs.append(changeingRec)
            #         dispMaps_morphed = torch.from_numpy(np.stack(dispMaps_morphed, axis=0)).unsqueeze(1).cuda()
            #         suboutputs['dispMaps_morphed'] = dispMaps_morphed
            #         changeingRecs = torch.from_numpy(np.stack(changeingRecs, axis=0)).unsqueeze(1).cuda()
            #
            #
            #         subBorderMorphLoss = torch.mean((suboutputs[('disp', 0)] - suboutputs[
            #             "dispMaps_morphed"]) ** 2 * self.tool.mask) * self.opt.borderMorphScale
            #
            #
            #         self.model_optimizer.zero_grad()
            #         subBorderMorphLoss.backward()
            #         self.model_optimizer.step()

            # outputs, losses = self.process_batch(inputs)
            # if self.opt.is_GAN_Training:
            #     self.set_requires_grad(self.models["encoder"], True)
            #     self.set_requires_grad(self.models["depth"], True)
            #     self.set_requires_grad(self.netD, False)
            #
            #     pred_fake = self.netD(outputs['fake_AB'])
            #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            #     # Second, G(A) = B
            #     self.loss_G_L1 = self.criterionL1(outputs['disp', 0], outputs['dispMaps_morphed']) * 100
            #     # combine loss and calculate gradients
            #     self.loss_G = self.loss_G_GAN + self.loss_G_L1
            #
            #     losses["totLoss"] = losses["totLoss"] + self.loss_G * 0.1

            # if is_normal_training:
            #     outputs, losses = self.supervised_with_photometric(inputs)
            # else:
            #     outputs, losses = self.supervised_with_morph(inputs)
                # outputs = dict()
                # losses = dict()
                # with open('filename.pickle', 'rb') as handle:
                #     unserialized_data = pickle.load(handle)
                #     outputs['dispMaps_morphed'] = unserialized_data.cuda()
                #
                # for key, ipt in inputs.items():
                #     if not (key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta'):
                #         inputs[key] = ipt.to(self.device)
                #
                # features = self.models["encoder"](inputs["color_aug", 0, 0])
                # outputs.update(self.models["depth"](features, computeSemantic=False, computeDepth=True))
                # self.merge_multDisp(inputs, outputs)
                # losses["totLoss"] = torch.mean((outputs['disp', 0] - outputs['dispMaps_morphed']) ** 2) * 100

            outputs, losses = self.supervised_with_photometric(inputs)
            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 1
            late_phase = self.step % self.opt.val_frequency == 0

            if early_phase or late_phase:

                if "loss_semantic" in losses:
                    loss_seman = losses["loss_semantic"].cpu().data
                    with torch.no_grad():
                        self.compute_semantic_losses(inputs, outputs, losses)
                else:
                    loss_seman = -1

                if "loss_depth" in losses:
                    loss_depth = losses["loss_depth"].cpu().data
                else:
                    loss_depth = -1

                self.log_time(batch_idx, duration, loss_seman, loss_depth, losses["totLoss"])

                if "depth_gt" in inputs and ('depth', 0, 0) in outputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses, writeImage=False)
                if self.step % self.opt.val_frequency == 0:
                    self.val()
                    if self.opt.writeImg:
                        if 'dispMaps_morphed' in outputs:
                            # self.record_img(disp = outputs['disp', 0], semantic_gt = inputs['seman_gt'], disp_morphed = outputs['dispMaps_morphed'], mask = outputs['grad_proj_msak'] * (1-outputs['ssimMask']))
                            self.record_img(disp=outputs['disp', 0], semantic_gt=inputs['seman_gt'],
                                            disp_morphed=outputs['dispMaps_morphed'],
                                            mask=outputs['selector_mask'])

                        else:
                            self.record_img(disp=outputs['disp', 0], semantic_gt=inputs['seman_gt'])
            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not(key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta' or key == 'file_add'):
                inputs[key] = ipt.to(self.device)
        tags = inputs['tag']
        banSemanticsFlag = 'kitti' in tags and not self.opt.predictboth
        # banDepthFlag = 'cityscape' in tags and not self.opt.predictboth
        banDepthFlag = 'cityscape' in tags
        all_color_aug = torch.cat([inputs[("color_aug", 0, 0)], inputs[("color_aug", 's', 0)]], dim=1)
        # tensor2rgb(inputs[("color_aug", 0, 0)], ind=0).show()
        # tensor2rgb(inputs[("color_aug", 's', 0)], ind=0).show()
        #
        # tensor2rgb(inputs[("color_aug", 0, 0)], ind=1).show()
        # tensor2rgb(inputs[("color_aug", 's', 0)], ind=1).show()
        #
        # tensor2rgb(inputs[("color_aug", 0, 0)], ind=2).show()
        # tensor2rgb(inputs[("color_aug", 's', 0)], ind=2).show()
        #
        # tensor2rgb(inputs[("color_aug", 0, 0)], ind=3).show()
        # tensor2rgb(inputs[("color_aug", 's', 0)], ind=3).show()
        features = self.models["encoder"](all_color_aug)
        outputs = dict()

        # for i in range(self.opt.batch_size):
        #     tensor2rgb(inputs["color_aug", 0, 0], ind= i).show()
        #     tensor2semantic(inputs['seman_gt'], ind=i).show()

        if not banSemanticsFlag:
            outputs.update(self.models["depth"](features, computeSemantic = True, computeDepth = False))
        if not banDepthFlag:
            outputs.update(self.models["depth"](features, computeSemantic = False, computeDepth = True))

        self.merge_multDisp(inputs, outputs)
        if not banDepthFlag:
            self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)
        return outputs, losses
    def is_regress_dispLoss(self, inputs, outputs):
        # if there are stereo images, we compute depth
        if ('color', 0, 0) in inputs and ('color', 's', 0) in inputs and ('disp', 0) in outputs and not "cityscape" in inputs['tag']:
            return True
        else:
            return False
    def is_regress_semanticLoss(self, inputs, outputs):
        # if there are semantic ground truth, we compute semantics
        if 'seman_gt' in inputs and ('seman', 0) in outputs:
            return True
        else:
            return False
    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if self.is_regress_dispLoss(inputs, outputs):
                self.compute_depth_losses(inputs, outputs, losses)
            if self.is_regress_semanticLoss(inputs, outputs):
                self.compute_semantic_losses(inputs, outputs, losses)
            self.log("val", inputs, outputs, losses, self.opt.writeImg)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        tag = inputs['tag'][0]
        height = inputs["height"][0]
        width = inputs["width"][0]
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(disp, [height, width], mode="bilinear", align_corners=False)
                source_scale = 0

            scaledDisp, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)

            outputs[("depth", 0, scale)] = depth

            frame_id = "s"
            T = inputs["stereo_T"]
            cam_points = self.backproject_depth[(tag, source_scale)](
                depth, inputs[("inv_K", source_scale)])
            pix_coords = self.project_3d[(tag, source_scale)](
                cam_points, inputs[("K", source_scale)], T)


            outputs[("sample", frame_id, scale)] = pix_coords
            if scale == 0:
                grad_proj_msak = (pix_coords[:,:,:,0] > -1) * (pix_coords[:,:,:,1] > -1) * (pix_coords[:, :, :, 0] < 1) * (pix_coords[:, :, :, 1] < 1)
                grad_proj_msak = grad_proj_msak.unsqueeze(1).float()
                outputs['grad_proj_msak'] = grad_proj_msak

            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[("color", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")

            outputs[("disp", scale)] = disp
            if scale == 0:
                outputs[("real_scale_disp", scale)] = scaledDisp * (torch.abs(inputs[("K", source_scale)][:, 0, 0] * T[:, 0, 3]).view(self.opt.batch_size, 1, 1, 1).expand_as(scaledDisp))
    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss

    def compute_arb_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.arbssim(pred, target).mean(1, True)
            reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss
    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        ssimLossMean = 0
        loss = 0

        if self.is_regress_dispLoss(inputs, outputs):
            source_scale = 0
            target = inputs[("color", 0, source_scale)]
            if self.opt.selfocclu:
                sourceSSIMMask = self.selfOccluMask(outputs[('real_scale_disp', source_scale)], inputs['stereo_T'][:, 0, 3])
                outputs['ssimMask'] = sourceSSIMMask
            if self.opt.de_flying_blob:
                sourceSSIMMask_stereo = self.selfOccluMask(outputs[('real_scale_disp', source_scale)], -inputs['stereo_T'][:, 0, 3])
                outputs['ssimMask_stereo'] = sourceSSIMMask_stereo
                # tensor2disp(sourceSSIMMask_stereo, vmax=0.08, ind=0).show()
                # tensor2disp(sourceSSIMMask, vmax=0.08, ind=0).show()
            if self.opt.seman_reg:
                skyMaks = (inputs['seman_gt'] == 10).float()

            if self.opt.lr_consistence:
                with torch.no_grad():
                    outputs_s = dict()
                    outputs_s.update(
                        self.models["depth"](self.models["encoder"](inputs['color_aug', 's', 0]), computeSemantic=False,
                                             computeDepth=True))
                    # outputs_s.update(
                    #     self.models["depth"](self.models["encoder"](torch.flip(inputs['color_aug', 0, 0], dims=[3])), computeSemantic=False,
                    #                          computeDepth=True))
                    # torch.flip(spec_mask[entry], dims=[1])
            for scale in self.opt.scales:
                reprojection_losses = []

                for frame_id in self.opt.frame_ids[1:]:
                    pred = outputs[("color", frame_id, scale)]
                    reprojection_losses.append(self.compute_reprojection_loss(pred, target))

                reprojection_losses = torch.cat(reprojection_losses, 1)

                if not self.opt.disable_automasking:
                    identity_reprojection_losses = []
                    for frame_id in self.opt.frame_ids[1:]:
                        pred = inputs[("color", frame_id, source_scale)]
                        identity_reprojection_losses.append(
                            self.compute_reprojection_loss(pred, target))

                    identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)

                    if self.opt.avg_reprojection:
                        identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                    else:
                        # save both images, and do min all at once below
                        identity_reprojection_loss = identity_reprojection_losses

                if self.opt.avg_reprojection:
                    reprojection_loss = reprojection_losses.mean(1, keepdim=True)
                else:
                    reprojection_loss = reprojection_losses

                if not self.opt.disable_automasking:
                    # add random numbers to break ties
                    identity_reprojection_loss += torch.randn(
                        identity_reprojection_loss.shape).cuda() * 0.00001

                    combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
                else:
                    combined = reprojection_loss

                if combined.shape[1] == 1:
                    to_optimise = combined
                else:
                    to_optimise, idxs = torch.min(combined, dim=1)

                if self.opt.read_stereo:
                    outputs[('to_optimize', scale)] = to_optimise

                if self.opt.selfocclu:
                    to_optimise = (1 - sourceSSIMMask.squeeze(1)) * to_optimise

                if ('mask', 0) in inputs:
                    if 'gtMask' in outputs:
                        andMask = outputs['gtMask'][:,0,:,:] * inputs[('mask', 0)]
                        to_optimise = to_optimise.masked_select(andMask)
                    else:
                        to_optimise = to_optimise.masked_select(inputs[('mask', 0)])
                else:
                    to_optimise = to_optimise
                ssimLoss = to_optimise.mean()
                loss += ssimLoss
                ssimLossMean += ssimLoss
                losses["loss_depth/{}".format(scale)] = ssimLoss

                if self.opt.disparity_smoothness > 1e-10 and not self.opt.secondOrderSmooth:
                    mult_disp = outputs[('disp', scale)]
                    mean_disp = mult_disp.mean(2, True).mean(3, True)
                    norm_disp = mult_disp / (mean_disp + 1e-7)
                    smooth_loss = get_smooth_loss(norm_disp, target)
                    if scale == 0:
                        losses["loss_reg/{}".format("smooth")] = smooth_loss
                    loss = loss + self.opt.disparity_smoothness * smooth_loss / (2 ** scale)

                if self.opt.secondOrderSmooth:
                    secOrder = self.compSecOrder.computegrad11(outputs[('disp', scale)])
                    secOrderLoss = torch.mean(secOrder)
                    loss = loss + 0.001 * self.opt.secondOrderSmoothScale * secOrderLoss / (2 ** scale)
                    # loss = secOrderLoss
                    if scale == 0:
                        losses["secoderSmooth"] = secOrderLoss

                if self.opt.borderMergeLoss:
                    if scale == 0:
                        if not self.opt.use_kitti_gt_semantics:
                            semantic_processed = torch.sum(self.sfx(outputs[('seman', 0)])[:, self.foregroundType, :, :], dim=1, keepdim=True)
                        else:
                            semantic_processed = torch.ones(self.opt.height, self.opt.width).cuda().byte()
                            for m in self.foregroundType:
                                semantic_processed = semantic_processed * (inputs['seman_gt'] != m)
                            semantic_processed = 1 - semantic_processed
                        # tensor2disp(semantic_processed, ind=0, vmax=1).show()
                        # tensor2disp(outputs[('disp', 0)], ind=0, vmax=0.11).show()
                        loss_borderMerge = self.borderMerge(outputs[('disp', 0)], semantic_processed.float())
                        # loss_borderMerge = self.borderMerge.forward3(outputs[('disp', 0)], semantic_processed.float())
                        # print("loss_borderMerge:%f" % loss_borderMerge)
                        # loss = loss + loss_borderMerge * 0.001
                        # loss = loss + loss_borderMerge * 10
                        loss = loss + loss_borderMerge * 10 * self.opt.borderMergeLossScale
                        if self.epoch > int(self.opt.num_epochs / 2):
                            self.opt.borderMergeLossScale = 0
                        # loss = loss_borderMerge
                        # loss = loss + loss_borderMerge * 0
                        losses["loss_bordermerge"] = loss_borderMerge
                    # self.borderMerge = BorderConverge(height, width, self.opt.batch_size).cuda()
                if self.opt.seman_reg:
                    skyReg = torch.mean(skyMaks * outputs[('disp', scale)])
                    if scale == 0:
                        losses['seman_reg/sky_reg'] = skyReg
                    loss = loss + skyReg * 1e-1

                if self.opt.lr_consistence:
                    with torch.no_grad():
                        sampled_disp = F.grid_sample(
                            outputs_s[('mul_disp', 0)],
                            outputs[("sample", frame_id, scale)],
                            padding_mode="border")
                        synthesize_mask = outputs['grad_proj_msak'] * (1 - outputs['ssimMask'])
                        sampled_disp = sampled_disp * synthesize_mask
                        # tensor2disp(sampled_disp, ind=0, vmax=0.08).show()
                        tensor2disp(outputs[('mul_disp', 0)], ind=0, vmax=0.08).show()
                        tensor2disp(outputs_s[('mul_disp', 0)], ind=0, vmax=0.08).show()
                        # tensor2disp(torch.flip(outputs_s[('mul_disp', 0)], dims=[3]), ind=0, vmax=0.08).show()
                        # tensor2disp(outputs_s[('mul_disp', 0)], ind=0, vmax=0.08).show()
                    lr_reg_loss = torch.sum(torch.abs(sampled_disp - outputs[('mul_disp', 0)]) * synthesize_mask) / torch.sum(synthesize_mask + 1)
                    if scale == 0:
                        losses['lr_reg'] = lr_reg_loss
                    loss = loss + lr_reg_loss * 1e-2
                    # a = torch.abs(sampled_disp - outputs[('mul_disp', 0)]) * synthesize_mask
                    # tensor2disp(a, ind=0, vmax=0.01).show()

                if self.opt.de_flying_blob:
                    # to_optimise_db = torch.mean(to_optimise_db * outputs['ssimMask_stereo'])
                    if (not self.opt.delay_open) or (self.epoch > 2):
                        to_optimise_db = torch.mean(to_optimise * outputs['ssimMask_stereo'])
                        loss = loss + to_optimise_db * self.opt.de_flying_blob_weight


                if self.opt.concrete_reg:
                    # start = timer()
                    if scale == 0:
                        penal_concrete = self.csp.get_3d_pts(depthmap=outputs[('depth', 0, scale)], intrinsic=inputs['realIn'], extrinsic=inputs['realEx'])
                        penal_concrete = penal_concrete[0,0,:,:].detach().cpu().numpy()
                        fig2 = tensor2rgb(inputs[('color', 0, 0)], ind=0)
                        fig2_arr = np.array(fig2)
                        fig2_arr[:,:,0][penal_concrete == 1] = 127 + fig2_arr[:,:,0][penal_concrete == 1] / 2
                        fig2_arr[:, :, 1][penal_concrete == 1] = fig2_arr[:,:,0][penal_concrete == 1] / 2
                        fig2_arr[:, :, 2][penal_concrete == 1] = fig2_arr[:,:,0][penal_concrete == 1] / 2
                        fig2 = pil.fromarray(fig2_arr)
                        fig2.save(os.path.join('/media/shengjie/other/sceneUnderstanding/SDNET/visualization/visualize_bad_pts', str(self.step) + '.png'))
                    # if scale == 0:
                    #     losses['loss_reg/concrete'] = penal_concrete
                    # loss = loss + penal_concrete * self.opt.concrete_reg_param
                    # end = timer()
                    # print(end - start)

                if False:
                    if scale == 0:
                        foregroundMapGt = torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width], dtype=torch.uint8, device=torch.device("cuda"))
                        for m in self.foregroundType:
                            foregroundMapGt = foregroundMapGt * (inputs['seman_gt'] != m)
                        foregroundMapGt = (1 - foregroundMapGt).float()
                        # skyMaks = (inputs['seman_gt'] == 10).float()

                        disparity_grad = torch.abs(self.tool.convDispx(outputs['disp', 0])) + torch.abs(self.tool.convDispy(outputs['disp', 0]))
                        semantics_grad = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(self.tool.convDispy(foregroundMapGt))
                        disparity_grad = disparity_grad * self.tool.zero_mask
                        semantics_grad = semantics_grad * self.tool.zero_mask

                        disparity_grad_bin = disparity_grad > self.tool.disparityTh
                        semantics_grad_bin = semantics_grad > self.tool.semanticsTh

                        # tensor2disp(disparity_grad_bin, ind=0, vmax=1).show()
                        # tensor2disp(semantics_grad_bin, ind=0, vmax=1).show()

                        disparity_grad_bin = disparity_grad_bin.detach().cpu().numpy()
                        semantics_grad_bin = semantics_grad_bin.detach().cpu().numpy()

                        disparityMap_to_processed = outputs['disp', 0].detach().cpu().numpy()
                        dispMaps_morphed = list()
                        changeingRecs = list()
                        for mm in range(self.opt.batch_size):
                            dispMap_morphed, changeingRec = self.auto_morph.automorph(disparity_grad_bin[mm, 0, :, :], semantics_grad_bin[mm, 0, :, :], disparityMap_to_processed[mm, 0, :, :])
                            dispMaps_morphed.append(dispMap_morphed)
                            changeingRecs.append(changeingRec)
                        dispMaps_morphed = torch.from_numpy(np.stack(dispMaps_morphed, axis=0)).unsqueeze(1).cuda()
                        outputs['dispMaps_morphed'] = dispMaps_morphed
                        changeingRecs = torch.from_numpy(np.stack(changeingRecs, axis=0)).unsqueeze(1).cuda()
                        # tensor2disp(dispMaps_morphed, ind=0, vmax=0.1).show()
                        # tensor2disp(dispMaps_morphed * self.tool.mask, ind=0, vmax=0.1).show()

                        subBorderMorphLoss = torch.mean(((outputs[('mul_disp', 0)] - outputs["dispMaps_morphed"]) * self.tool.mask) ** 2) * self.opt.borderMorphScale
                        # loss_borderMorph = torch.sum((torch.abs(dispMaps_morphed - outputs['disp', 0]) * self.tool.mask)**2) / (torch.sum(changeingRecs > 0) + 1) * 1e3
                        # loss = loss + loss_borderMorph * self.opt.borderMorphScale + torch.sum(outputs['disp', 0] * skyMaks) / (torch.sum(skyMaks) + 1) * float((self.opt.borderMorphScale > 0))
                        # loss = loss + loss_borderMorph * self.opt.borderMorphScale
                        losses["loss_borderMorph"] = subBorderMorphLoss

                        # with open('filename.pickle', 'rb') as handle:
                        #     unserialized_data = pickle.load(handle)
                        #     outputs['dispMaps_morphed'] = unserialized_data.cuda()
                        #
                        # if self.opt.is_GAN_Training:
                        #     if backD:
                        #         self.set_requires_grad(self.models["encoder"], False)
                        #         self.set_requires_grad(self.models["depth"], False)
                        #         self.set_requires_grad(self.netD, True)
                        #
                        #         self.optimizer_D.zero_grad()
                        #         fake_AB = torch.cat((outputs['disp', 0], foregroundMapGt),
                        #                             1)  # we use conditional GANs; we need to feed both input and output to the discriminator
                        #         pred_fake = self.netD(fake_AB.detach())
                        #         self.loss_D_fake = self.criterionGAN(pred_fake, False)
                        #
                        #         real_AB = torch.cat((outputs['dispMaps_morphed'], foregroundMapGt), 1)
                        #         pred_real = self.netD(real_AB.detach())
                        #         self.loss_D_real = self.criterionGAN(pred_real, True)
                        #         self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
                        #
                        #         outputs['fake_AB'] = fake_AB
                        #         outputs['real_AB'] = real_AB
                        #
                        #         self.loss_D.backward()
                        #         self.optimizer_D.step()

            loss = loss / self.num_scales
            # with open('filename.pickle', 'wb') as handle:
            #     pickle.dump(outputs['dispMaps_morphed'], handle, protocol=pickle.HIGHEST_PROTOCOL)
            # with open('filename.pickle', 'rb') as handle:
            #     unserialized_data = pickle.load(handle)
            # dispMaps_morphed = unserialized_data.cuda()
            # outputs['dispMaps_morphed'] = dispMaps_morphed
            # loss = torch.mean((outputs['disp', 0] - dispMaps_morphed)**2) * 1e2

            # diff = torch.abs(dispMaps_morphed - outputs['disp', 0])
            # tensor2disp(diff, ind=0, vmax=0.07).show()
            losses["loss_depth"] = ssimLossMean / self.num_scales

        if self.is_regress_semanticLoss(inputs, outputs):
            loss_seman, loss_semantoshow = self.semanticLoss(inputs, outputs) # semantic loss is scaled already
            for entry in loss_semantoshow:
                losses[entry] = loss_semantoshow[entry]
            loss = loss + self.semanticCoeff * loss_seman
            losses["loss_semantic"] = loss_seman

        losses["totLoss"] = loss
        return losses



    def compute_semantic_losses(self, inputs, outputs, losses):
        """Compute semantic metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        gt = inputs['seman_gt_eval'].cpu().numpy().astype(np.uint8)
        pred = self.sfx(outputs[('seman', 0)]).detach()
        pred = torch.argmax(pred, dim=1).type(torch.float).unsqueeze(1)
        pred = F.interpolate(pred, [gt.shape[1], gt.shape[2]], mode='nearest')
        pred = pred.squeeze(1).cpu().numpy().astype(np.uint8)
        # visualize_semantic(gt[0,:,:]).show()
        # visualize_semantic(pred[0,:,:]).show()

        confMatrix = generateMatrix(args)
        groundTruthNp = gt
        predictionNp = pred
        # imgWidth = groundTruthNp.shape[1]
        # imgHeight = groundTruthNp.shape[0]
        nbPixels = groundTruthNp.shape[0] * groundTruthNp.shape[1] * groundTruthNp.shape[2]

        # encoding_value = max(groundTruthNp.max(), predictionNp.max()).astype(np.int32) + 1
        encoding_value = 256 # precomputed
        encoded = (groundTruthNp.astype(np.int32) * encoding_value) + predictionNp

        values, cnt = np.unique(encoded, return_counts=True)
        count255 = 0
        for value, c in zip(values, cnt):
            pred_id = value % encoding_value
            gt_id = int((value - pred_id) / encoding_value)
            if pred_id == 255 or gt_id == 255:
                count255 = count255 + c
                continue
            if not gt_id in args.evalLabels:
                printError("Unknown label with id {:}".format(gt_id))
            confMatrix[gt_id][pred_id] += c

        if confMatrix.sum() +  count255!= nbPixels:
            printError(
                'Number of analyzed pixels and entries in confusion matrix disagree: contMatrix {}, pixels {}'.format(
                    confMatrix.sum(), nbPixels))

        classScoreList = {}
        for label in args.evalLabels:
            labelName = trainId2label[label].name
            classScoreList[labelName] = getIouScoreForLabel(label, confMatrix, args)
        vals = np.array(list(classScoreList.values()))
        losses['mIOU'] = np.mean(vals[np.logical_not(np.isnan(vals))])

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss_semantic, loss_depth, loss_tot):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss_semantic: {:.5f} | loss_depth: {:.5f} | loss_tot: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss_semantic, loss_depth, loss_tot,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def record_img(self, disp, semantic_gt, disp_morphed = None, mask = None):
        dirpath = os.path.join("/media/shengjie/other/sceneUnderstanding/SDNET/visualization", self.opt.model_name)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        viewIndex = 0
        fig_seman = tensor2semantic(semantic_gt, ind=viewIndex, isGt=True)
        fig_disp = tensor2disp(disp, ind=viewIndex, vmax=0.09)
        overlay_org = pil.fromarray((np.array(fig_disp) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))

        # plt.figure()
        # plt.imshow(overlay_org)
        #
        # mask_np = np.zeros_like(np.array(overlay_org))
        # mask_np[120:160, 220:252] = 1
        # overlay_org = pil.fromarray(np.array(overlay_org) * mask_np)
        # plt.figure()
        # plt.imshow(overlay_org)

        if disp_morphed is not None:
            fig_disp_morphed = tensor2disp(disp_morphed, ind=viewIndex, vmax=0.09)
            overlay_dst = pil.fromarray((np.array(fig_disp_morphed) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))

            disp_masked = disp * mask
            fig_disp_masked = tensor2disp(disp_masked, vmax=0.09, ind=viewIndex)
            fig_disp_masked_overlay = pil.fromarray((np.array(fig_disp_masked) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))
            # mask_np = mask[viewIndex, 0, :, :].detach().cpu().numpy()
            # tensor2disp(mask, vmax=1, ind=0).show()
            # fig_disp = pil.fromarray((np.array(fig_disp) * np.expand_dims(mask_np, 2).repeat(3, 2)).astype(np.uint8))


            combined_fig = pil.fromarray(np.concatenate(
                [np.array(overlay_org), np.array(fig_disp), np.array(overlay_dst), np.array(fig_disp_morphed), np.array(fig_disp_masked_overlay)], axis=0))
        else:
            combined_fig = pil.fromarray(np.concatenate(
                [np.array(overlay_org), np.array(fig_disp)], axis=0))
        combined_fig.save(
            dirpath + '/' + str(self.step) + ".png")

    def log(self, mode, inputs, outputs, losses, writeImage = False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if l != 'totLoss':
                writer.add_scalar("{}".format(l), v, self.step)

        # if writeImage:
            # viewInd = 1
            # cm = plt.get_cmap('magma')
            # dispimg = outputs[("disp", 0)][viewInd,0,:,:].cpu().numpy()
            # dispimg = dispimg / 0.07
            # viewmask = (cm(dispimg) * 255).astype(np.uint8)
            #
            # slice = inputs['seman_gt'][viewInd, 0, :, :].cpu().numpy()
            # seman = visualize_semantic(slice)
            # overlay = (np.array(viewmask[:,:,0:3]) * 0.7 + 0.3 * np.array(seman)).astype(np.uint8)
            # if self.opt.selfocclu:
            #     dispSuppressed = (outputs[("disp", 0)] * (1 - outputs['ssimMask']))[viewInd,0,:,:].cpu().numpy()
            #     dispSuppressed = dispSuppressed / 0.1
            #     viewSupmask = (cm(dispSuppressed) * 255).astype(np.uint8)
            #     supoverlay = (np.array(viewSupmask[:, :, 0:3]) * 0.7 + 0.3 * np.array(seman)).astype(np.uint8)
            #     overlay = np.concatenate([overlay, viewmask[:, :, 0:3], viewSupmask[:, :, 0:3], supoverlay[:,:,0:3]], axis=0)
            # else:
            #     overlay = np.concatenate([overlay, viewmask[:, :, 0:3]],
            #                              axis=0)
            #
            # fig_predict = tensor2rgb(outputs[('color', 's', 0)], ind=0)
            # fig_target = tensor2rgb(inputs[('color', 0, 0)], ind=0)
            # fig_source = tensor2rgb(inputs[('color', 's', 0)], ind=0)
            # fig_combined = pil.fromarray(np.concatenate([np.array(fig_predict), np.array(fig_target), np.array(fig_source)], axis=0))
            # fig_combined.save("/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/trianReCompare/" + str(self.step) + ".png")


            # viewInd = 0
            # semanMapEst = outputs[('seman', 0)]
            # semanMapEst_sfxed = self.sfx(semanMapEst)
            # semanMapEst_inds = torch.argmax(semanMapEst_sfxed, dim=1).unsqueeze(1)
            # seman_est_fig = tensor2semantic(semanMapEst_inds, ind=0)
            # dispimg = outputs[("disp", 0)]
            # dispimg_fig = tensor2disp(dispimg, ind=viewInd, vmax=0.08)
            # combined = np.concatenate([np.array(seman_est_fig), np.array(dispimg_fig)[:,:,0:3]], axis=0)
            # semantic_processed = torch.ones(self.opt.height, self.opt.width).cuda().byte()
            # for m in self.foregroundType:
            #     semantic_processed = semantic_processed * (inputs['seman_gt'] != m)
            # semantic_processed = 1 - semantic_processed
            # fig_rgb = self.borderMerge.visualizeForDebug(outputs[('disp', 0)], semantic_processed.float())
            # combined = np.concatenate([combined, np.array(fig_rgb)[:,:,0:3]], axis=0)
            # pil.fromarray(combined).save("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderMerge/" + str(self.step) + ".png")

            # foregroundMapGt = torch.ones([self.opt.batch_size, 1, self.opt.height, self.opt.width],
            #                              dtype=torch.uint8, device=torch.device("cuda"))
            # for m in self.foregroundType:
            #     foregroundMapGt = foregroundMapGt * (inputs['seman_gt'] != m)
            # foregroundMapGt = (1 - foregroundMapGt).float()
            #
            # disparity_grad = torch.abs(self.tool.convDispx(outputs['disp', 0])) + torch.abs(
            #     self.tool.convDispy(outputs['disp', 0]))
            # semantics_grad = torch.abs(self.tool.convDispx(foregroundMapGt)) + torch.abs(
            #     self.tool.convDispy(foregroundMapGt))
            # disparity_grad = disparity_grad * self.tool.zero_mask
            # semantics_grad = semantics_grad * self.tool.zero_mask
            #
            # disparity_grad_bin = disparity_grad > self.tool.disparityTh
            # semantics_grad_bin = semantics_grad > self.tool.semanticsTh
            #
            # disparity_grad_bin = disparity_grad_bin.detach().cpu().numpy()
            # semantics_grad_bin = semantics_grad_bin.detach().cpu().numpy()
            #
            # disparityMap_to_processed = outputs['disp', 0].detach().cpu().numpy()
            # dispMaps_morphed = list()
            # changeingRecs = list()
            # for mm in range(self.opt.batch_size):
            #     dispMap_morphed, changeingRec = self.auto_morph.automorph(disparity_grad_bin[mm, 0, :, :],
            #                                                               semantics_grad_bin[mm, 0, :, :],
            #                                                               disparityMap_to_processed[mm, 0, :, :])
            #     dispMaps_morphed.append(dispMap_morphed)
            #     changeingRecs.append(changeingRec)
            # dispMaps_morphed = torch.from_numpy(np.stack(dispMaps_morphed, axis=0)).unsqueeze(1).cuda()
            # outputs['dispMaps_morphed'] = dispMaps_morphed
            #



            # viewIndex = 0
            # fig_seman = tensor2semantic(inputs['seman_gt'], ind=viewIndex, isGt=True)
            # fig_disp = tensor2disp(outputs[('disp', 0)], ind=viewIndex, vmax=0.09)
            # overlay_org = pil.fromarray((np.array(fig_disp) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))
            # if 'dispMaps_morphed' in outputs:
            #     fig_disp_morphed = tensor2disp(outputs['dispMaps_morphed'], ind=viewIndex, vmax=0.09)
            #     overlay_dst = pil.fromarray((np.array(fig_disp_morphed) * 0.7 + np.array(fig_seman) * 0.3).astype(np.uint8))
            #     combined_fig = pil.fromarray(np.concatenate([np.array(overlay_org), np.array(fig_disp), np.array(overlay_dst), np.array(fig_disp_morphed)],axis=0))
            # else:
            #     combined_fig = pil.fromarray(np.concatenate(
            #         [np.array(overlay_org), np.array(fig_disp)], axis=0))
            # combined_fig.save("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderMerge/" + str(self.step) + ".png")



    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, indentifier = None):
        """Save model weights to disk
        """
        if indentifier is None:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        else:
            save_folder = os.path.join(self.log_path, "models", "weights_{}".format(indentifier))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            if model_name == 'stable_encoder' or  model_name == 'stable_depth':
                continue
            to_save = model.state_dict()
            if model_name == 'encoder':
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)
        # if self.opt.is_GAN_Training:
        #     torch.save(self.netD.state_dict(), os.path.join(save_folder, "{}.pth".format("netD")))

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)
        print("save to %s" % save_folder)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)
        # if self.opt.is_GAN_Training:
        #     if os.path.isfile(os.path.join(self.opt.load_weights_folder, "{}.pth".format("netD"))):
        #         self.netD.state_dict().update(torch.load(os.path.join(self.opt.load_weights_folder, "{}.pth".format("netD"))))

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad