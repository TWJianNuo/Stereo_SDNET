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
import networks.DenseASPP

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
# from IPython import embed
from utils import my_Sampler
import cityscapesscripts.helpers.labels
from cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import *
# import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

torch.manual_seed(0)

class Trainer_ASPP:
    def __init__(self, options):
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

        self.Model_CFG = {
            'bn_size': 4,
            'drop_rate': 0,
            'growth_rate': 32,
            'num_init_features': 64,
            'block_config': (6, 12, 24, 16),

            'dropout0': 0.1,
            'dropout1': 0.1,
            'd_feature0': 128,
            'd_feature1': 64,
        }
        self.N_CLASS = 19
        self.models["depth"] = networks.DenseASPP(self.Model_CFG, n_class=self.N_CLASS, output_stride=8)
        self.models["depth"].to(self.device)

        # self.models["encoder"] = networks.ResnetEncoder(
        #     self.opt.num_layers, self.opt.weights_init == "pretrained")
        # self.models["encoder"].to(self.device)
        # self.parameters_to_train += list(self.models["encoder"].parameters())
        # self.models["depth"] = networks.DepthDecoder(
        #     self.models["encoder"].num_ch_enc, self.opt.scales, isSwitch=self.switchMode, isMulChannel=self.opt.isMulChannel)
        # self.models["depth"].to(self.device)

        self.parameters_to_train += list(self.models["depth"].parameters())
        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate, weight_decay=0.00001)
        # lr_lambda = torch.Tensor(np.power(1 - np.arange(self.opt.num_epochs) / self.opt.num_epochs, 0.9))
        # lr_lambda = np.power(1 - np.arange(self.opt.num_epochs) / self.opt.num_epochs, 0.9)
        lambda1 = lambda epoch, max_epoch = self.opt.num_epochs: np.power(1 - epoch / max_epoch, 0.9)
        self.model_lr_scheduler = optim.lr_scheduler.LambdaLR(self.model_optimizer, lr_lambda = [lambda1])

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        self.set_dataset()
        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

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

        self.save_opts()

        # self.timeSpan_decoder = 0
        # self.timeSpan_mergeLayer = 0
        # self.timeSpan_predict = 0
        # self.timeSpan_loss = 0
        # self.timeSpan_dispSmooth = 0
    def set_layers(self):
        """properly handle layer initialization under multiple dataset situation
        """
        self.semanticLoss = Compute_SemanticLoss(min_scale = np.array(self.opt.scales).max())
        self.merge_multDisp = Merge_MultDisp(self.opt.scales, batchSize = self.opt.batch_size, isMulChannel = self.opt.isMulChannel)
        self.compsurfnorm = {}
        self.backproject_depth = {}
        self.project_3d = {}
        if self.opt.selfocclu:
            self.selfOccluMask = SelfOccluMask().cuda()
            self.dispupLoss = ComputeDispUpLoss().cuda()
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

        if self.opt.mulReg:
            self.varWindow = 5
            for p, tag in enumerate(tags):
                height = self.format[p][1]
                width = self.format[p][2]
                self.compsurfnorm[tag] = ComputeSurfaceNormal(height = height, width = width, batch_size = self.opt.batch_size).cuda()
            self.objReg = ObjRegularization(windowsize=self.varWindow).cuda()
            self.skyId = name2label['sky'].trainId # sky
            self.wallType = [name2label['building'].trainId, name2label['wall'].trainId, name2label['fence'].trainId]  # Building, wall, fence
            self.roadType = [name2label['road'].trainId, name2label['sidewalk'].trainId, name2label['terrain'].trainId]  # road, sidewalk, terrain
            self.permuType = [name2label['pole'].trainId, name2label['traffic sign'].trainId]  # Pole, traffic sign

        if self.opt.borderRegression:
            self.foregroundType = [name2label['pole'].trainId, name2label['traffic light'].trainId, name2label['traffic sign'].trainId, name2label['person'].trainId,
                                   name2label['rider'].trainId, name2label['car'].trainId, name2label['truck'].trainId, name2label['bus'].trainId,
                                   name2label['train'].trainId, name2label['motorcycle'].trainId, name2label['bicycle'].trainId]
            # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
            self.backgroundType = [name2label['road'].trainId, name2label['sidewalk'].trainId, name2label['building'].trainId, name2label['wall'].trainId,
                                   name2label['fence'].trainId, name2label['vegetation'].trainId, name2label['terrain'].trainId, name2label['sky'].trainId]
            # road, sidewalk, building, wall, fence, vegetation, terrain, sky
            self.suppressType = [name2label['unlabeled'].trainId]  # Suppress no label lines
            self.borderRegress = BorderRegression()
            self.borderRegress.cuda()

        if self.opt.borderSim:
            self.foregroundType = [name2label['pole'].trainId, name2label['traffic light'].trainId, name2label['traffic sign'].trainId, name2label['person'].trainId,
                                   name2label['rider'].trainId, name2label['car'].trainId, name2label['truck'].trainId, name2label['bus'].trainId,
                                   name2label['train'].trainId, name2label['motorcycle'].trainId, name2label['bicycle'].trainId]
            self.rdSampleOnBorder = RandomSampleNeighbourPts(batchNum=self.opt.batch_size)
            self.rdSampleOnBorder.cuda()


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

            train_dataset = initFunc(
                datapath_set[i], train_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, tag=dataset_set[i], is_train=True, img_ext=img_ext, is_sep_train_seman = self.opt.is_sep_train_seman)
            train_sample_num[i] = train_dataset.__len__()
            stacked_train_datasets.append(train_dataset)

            val_dataset = initFunc(
                datapath_set[i], val_filenames, self.opt.height, self.opt.width,
                self.opt.frame_ids, 4, tag=dataset_set[i], is_train=False, img_ext=img_ext, is_sep_train_seman = self.opt.is_sep_train_seman)
            val_sample_num[i] = val_dataset.__len__()
            stacked_val_datasets.append(val_dataset)

        initFunc = datasets_dict['joint']
        self.joint_dataset_train = initFunc(stacked_train_datasets)
        joint_dataset_val = initFunc(stacked_val_datasets)

        self.trainSample = my_Sampler(train_sample_num, self.opt.batch_size) # train sampler is used for multi-stage training
        valSample = my_Sampler(val_sample_num, self.opt.batch_size)

        self.train_loader = DataLoader(
            self.joint_dataset_train, self.opt.batch_size, shuffle=False, sampler=self.trainSample,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_loader = DataLoader(
            joint_dataset_val, self.opt.batch_size, shuffle=False, sampler=valSample,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)
        self.train_iter = iter(self.train_loader)

        num_train_samples = self.joint_dataset_train.__len__()
        self.train_num = self.joint_dataset_train.__len__()
        self.val_num = joint_dataset_val.__len__()
        self.format = self.joint_dataset_train.format
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        if not self.is_use_sep_seman_train():
            for m in self.models.values():
                m.eval()
        else:
            for m in self.models.values():
                m.train()

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

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()
        print("Training")
        self.set_train()
        # adjust by changing the sampler
        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["totLoss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 1
            late_phase = self.step % 1 == 0

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
                # print("border loss1 %f, border loss % f" % (losses["loss_reg/borderReg"], losses["loss_reg/borderReg_overflow"]))
                if self.opt.borderRegression:
                    print("border loss %f" % (losses["loss_reg/borderReg"]))

                # if self.opt.borderSim and self.opt.debug:
                #     print("borderSim loss %f, borderContrast loss %f" % (losses["loss_reg/borderSimilar"], losses["loss_reg/borderContrast"]))

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses, writeImage=False)
                if self.step % 10 == 0:
                    self.val()

            # timeCount = np.array((self.timeSpan_decoder, self.timeSpan_mergeLayer, self.timeSpan_predict, self.timeSpan_loss))
            # timeCount = timeCount / np.sum(timeCount)
            # print("Time util: | decoder:%f | mergeLayer:%f | prediction:%f | loss:%f | smoothLoss:%f | weighRatio:%f |" % (timeCount[0], timeCount[1], timeCount[2], timeCount[3], self.timeSpan_dispSmooth/self.timeSpan_loss, self.merge_multDisp.weights_time / self.timeSpan_mergeLayer))
            self.step += 1


    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            if not(key == 'height' or key == 'width' or key == 'tag'):
                inputs[key] = ipt.to(self.device)

        # features = self.models["encoder"](inputs["color_aug", 0, 0])
        # just for check
        """
        i = 1
        img = pil.fromarray((inputs[("color_aug", 0, 0)].permute(0,2,3,1)[i,:,:,:].cpu().numpy() * 255).astype(np.uint8))
        img.show()
        label = inputs['seman_gt'].permute(0,2,3,1)[i,:,:,0].cpu().numpy()
        visualize_semantic(label).show()
        """

        # start_decoder = torch.cuda.Event(enable_timing=True)
        # end_decoder = torch.cuda.Event(enable_timing=True)
        # Switch between semantic and depth estimation
        # start_decoder.record()

        # for i in range(self.opt.batch_size):
            # semanlabbel = inputs['seman_gt'][i, 0, :, :].cpu().numpy().astype(np.uint8)
            # visualize_semantic(semanlabbel).show()

            # semanlabbel = inputs['seperate_seman_gt'][i, 0, :, :].cpu().numpy().astype(np.uint8)
            # visualize_semantic(semanlabbel).show()

            # img = pil.fromarray(
            #     (inputs[("seperate_seman_rgb")].permute(0, 2, 3, 1)[i, :, :, :].cpu().numpy() * 255).astype(np.uint8))
            # img.show()

            # img = pil.fromarray(
            #     (inputs[('color', 0, 0)].permute(0, 2, 3, 1)[i, :, :, :].cpu().numpy() * 255).astype(np.uint8))
            # img.show()

            # img = pil.fromarray(
            #     (inputs[('color', 's', 0)].permute(0, 2, 3, 1)[i, :, :, :].cpu().numpy() * 255).astype(np.uint8))
            # img.show()

            # img = pil.fromarray(
            #     (inputs[('color_aug', 0, 0)].permute(0, 2, 3, 1)[i, :, :, :].cpu().numpy() * 255).astype(np.uint8))
            # img.show()

            # img = pil.fromarray(
            #     (inputs[('color_aug', 's', 0)].permute(0, 2, 3, 1)[i, :, :, :].cpu().numpy() * 255).astype(np.uint8))
            # img.show()


        outputs = dict()
        if not self.opt.banSemantic:
            if not self.is_use_sep_seman_train():
                outputs.update(self.models["depth"](inputs["color_aug", 0, 0], computeSemantic = True, computeDepth = False))
            else:
                outputs.update(self.models["depth"](inputs['seperate_seman_rgb'], computeSemantic=True, computeDepth=False))
        if not self.opt.banDepth:
            outputs.update(self.models["depth"](inputs["color_aug", 0, 0], computeSemantic = False, computeDepth = True))


        self.merge_multDisp(inputs, outputs)
        if not self.opt.banDepth:
            self.generate_images_pred(inputs, outputs)
        losses = self.compute_losses(inputs, outputs)

        #-1.8883
        # check
        # inputs = self.val_iter.next()
        # for key, ipt in inputs.items():
        #     if not(key == 'height' or key == 'width' or key == 'tag'):
        #         inputs[key] = ipt.to(self.device)
        # outputs = dict()
        # if not self.opt.banSemantic:
        #     outputs.update(self.models["depth"](inputs["color_aug", 0, 0], computeSemantic = True, computeDepth = False))
        # gt = inputs['seman_gt_eval'].cpu().numpy().astype(np.uint8)
        # pred = self.sfx(outputs[('seman', 0)]).detach()
        # pred = torch.argmax(pred, dim=1).type(torch.float).unsqueeze(1)
        # pred = F.interpolate(pred, [gt.shape[1], gt.shape[2]], mode='nearest')
        # pred = pred.squeeze(1).cpu().numpy().astype(np.uint8)
        # visualize_semantic(gt[0, :, :]).show()
        # visualize_semantic(pred[0, :, :]).show()
        return outputs, losses
    def is_use_sep_seman_train(self):
        if self.opt.num_epochs - self.epoch <= 20 or not self.opt.is_sep_train_seman:
            # print("Not use")
            return False
        else:
            return True

    def is_regress_dispLoss(self, inputs, outputs):
        # if there are stereo images, we compute depth
        if ('color', 0, 0) in inputs and ('color', 's', 0) in inputs and ('disp', 0) in outputs:
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

            if "depth_gt" in inputs and ('depth', 0, 0) in outputs:
                self.compute_depth_losses(inputs, outputs, losses)
            if 'seman_gt_eval' in inputs and ('seman', 0) in outputs:
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

            outputs[("color", frame_id, scale)] = F.grid_sample(
                inputs[("color", frame_id, source_scale)],
                outputs[("sample", frame_id, scale)],
                padding_mode="border")

            outputs[("disp", scale)] = disp
            # if self.opt.isMulReg:
            #     outputs['surfacenorm'] = self.compsurfnorm[tag](depth * self.STEREO_SCALE_FACTOR)

            # for i, frame_id in enumerate(self.opt.frame_ids[1:]):
            #     if frame_id == "s":
            #         T = inputs["stereo_T"]
            #     else:
            #         T = outputs[("cam_T_cam", 0, frame_id)]
            #     cam_points = self.backproject_depth[(tag, source_scale)](
            #         depth, inputs[("inv_K", source_scale)])
            #     pix_coords = self.project_3d[(tag, source_scale)](
            #         cam_points, inputs[("K", source_scale)], T)
            #     outputs[("sample", frame_id, scale)] = pix_coords
            #     outputs[("color", frame_id, scale)] = F.grid_sample(
            #         inputs[("color", frame_id, source_scale)],
            #         outputs[("sample", frame_id, scale)],
            #         padding_mode="border")

            # Check:
            # depth = T[0,0,3] * inputs[("K", source_scale)][0,0,0] / scaledDisp
            # ones = nn.Parameter(torch.ones(10, 1, 256 * 512)).cuda()
            # inv_K = inputs[("inv_K", source_scale)]
            # org_pix_coords = self.backproject_depth[(tag, source_scale)].pix_coords
            # cam_points = torch.matmul(inv_K[:, :3, :3], org_pix_coords)
            # cam_points = depth.view(10, 1, -1) * cam_points
            # cam_points = torch.cat([cam_points, ones], 1)
            #
            # org_pix_coords = org_pix_coords[:,0:2,:]
            #
            # K = inputs[("K", source_scale)]
            # T = T
            # P = torch.matmul(K, T)[:, :3, :]
            # cam_points = torch.matmul(P, cam_points)
            # eps = 1e-7
            # pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + eps)
            # bias = (pix_coords - org_pix_coords)[:,0,:]
            # ratio = bias / scaledDisp[:,0,:,:].view(10,-1)
            # var = torch.mean(torch.abs(bias - scaledDisp[:,0,:,:].view(10,-1)))
            #
            # visualize_outpu(inputs, outputs, '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/recon_rg_img/kitti', np.random.randint(0, 100000, 1)[0])
            # if not self.opt.disable_automasking:
            #     outputs[("color_identity", frame_id, scale)] = \
            #         inputs[("color", frame_id, source_scale)]

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

    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        ssimLossMean = 0
        tag = inputs['tag'][0]
        loss = 0
        if self.is_regress_dispLoss(inputs, outputs):
            source_scale = 0
            target = inputs[("color", 0, source_scale)]
            # height = target.shape[2]
            # width = target.shape[3]
            if self.opt.selfocclu:
                # expand = torch.nn.MaxPool2d(kernel_size=5, padding=2, stride=1).cuda()
                sourceSSIMMask = self.selfOccluMask(outputs[('disp', source_scale)], inputs['stereo_T'][:,0,3])
                outputs['ssimMask'] = sourceSSIMMask
                # sourceSSIMMask = expand(sourceSSIMMask)
                # Check
                # viewind = 0
                # cm = plt.get_cmap('magma')
                # viewSSIMMask = sourceSSIMMask[viewind, 0, :, :].detach().cpu().numpy()
                # vmax = np.percentile(viewSSIMMask, 95)
                # viewSSIMMask = (cm(viewSSIMMask / vmax) * 255).astype(np.uint8)
                # pil.fromarray(viewSSIMMask).show()
                # viewSSIMMask, viewSSIMMask_opp = self.selfOccluMask.visualize(outputs[('disp', source_scale)], viewind=0)

            for scale in self.opt.scales:
                reprojection_losses = []
                # disp = outputs[("disp", scale)]
                # color = inputs[("color", 0, scale)]

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


                """
                # Check
                viewInd = 1
                errormap = reprojection_loss[viewInd, 0, :, :].detach().cpu().numpy()
                vmax = np.percentile(errormap, 90)
                slice = errormap / vmax
                cm = plt.get_cmap('magma')
                slice = (cm(slice) * 255).astype(np.uint8)

                pred = outputs[("color", frame_id, scale)]
                pred = pred[viewInd, :, :, :].detach().cpu().permute(1,2,0).numpy()
                pred = (pred * 255).astype(np.uint8)

                org = inputs[("color_aug", 0, source_scale)]
                org = org[viewInd, :, :, :].detach().cpu().permute(1,2,0).numpy()
                org = (org * 255).astype(np.uint8)

                andMask = outputs['gtMask'][:, 0, :, :] * inputs[('mask', 0)]
                andMask = andMask[0,:,:].cpu().numpy()
                masked_errormap = errormap * andMask
                masked_errormap = masked_errormap / vmax
                masked_errormap = (cm(masked_errormap) * 255).astype(np.uint8)

                andMask = outputs['gtMask'][:, 0, :, :] * inputs[('mask', 0)]
                andMask = andMask[0,:,:].cpu().numpy()
                andMask = np.expand_dims((andMask * 255).astype(np.uint8), 2)
                andMask = np.repeat(andMask, 3, 2)

                combined = np.concatenate([slice[:,:,0:3],masked_errormap[:,:,0:3], pred, org, andMask], axis=1)
                pil.fromarray(combined).show()

                muldisp = outputs[('mul_disp', 0)][viewInd,:,:,:]
                disp = outputs[('disp', 0)][viewInd,0,:,:].detach().cpu().numpy() / 0.2
                gtsemanMap = inputs['seman_gt'][viewInd,0,:,:].detach().cpu().numpy()
                for k in range(muldisp.shape[0] - 1):
                    newMask = gtsemanMap == k
                    if np.sum(newMask) < 100:
                        continue
                    dispmap = muldisp[k, :, :].detach().cpu().numpy() / 0.2
                    dispmap_spec = dispmap * newMask
                    dispmap = (cm(disp) * 255).astype(np.uint8)
                    dispmap_spec = (cm(dispmap_spec) * 255).astype(np.uint8)
                    edgeCombine = np.concatenate([dispmap_spec[:,:,0:3],dispmap[:,:,0:3]], axis=1)
                    pil.fromarray(edgeCombine).show()
                """


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

                if self.opt.selfocclu:
                    to_optimise = (1 - sourceSSIMMask.squeeze(1)) * to_optimise
                    # a, b = self.selfOccluMask.visualize(outputs[('disp', scale)])
                    # scaleSSIMMask = self.selfOccluMask(outputs[('disp', scale)])
                    # dispupMap = self.dispupLoss(outputs[('disp', scale)])
                    # dispupLoss = torch.mean(scaleSSIMMask * dispupMap) * self.opt.selfoccluscale
                    # losses["loss_dispup/{}".format(scale)] = dispupLoss
                    # loss += dispupLoss
                    # maskfig, dispfig = self.selfOccluMask.visualize(outputs[('disp', scale)])

                # if it's cityscape dataset, need to mask out ego vehicle
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

                if self.opt.mulReg and ('seman_gt' in inputs or ('seman', source_scale) in outputs):
                    surnormMap = self.compsurfnorm[tag](depthMap = outputs[('depth', 0, scale)], invcamK = inputs['invcamK'])

                    skyMask = inputs['seman_gt'] == self.skyId
                    skyerr = self.objReg.regularizeSky(outputs[('depth', 0, scale)], skyMask)

                    wallMask = torch.ones(inputs['seman_gt'].shape, dtype=torch.uint8).cuda()
                    roadMask = torch.ones(inputs['seman_gt'].shape, dtype=torch.uint8).cuda()
                    permuMask = torch.ones(inputs['seman_gt'].shape, dtype=torch.uint8).cuda()

                    with torch.no_grad():
                        for m in self.wallType:
                            wallMask = wallMask * (inputs['seman_gt'] != m)
                        wallMask = 1 - wallMask
                        wallMask = wallMask[:,:,1:-1,1:-1]

                        for m in self.roadType:
                            roadMask = roadMask * (inputs['seman_gt'] != m)
                        roadMask = 1 - roadMask
                        roadMask = roadMask[:,:,1:-1,1:-1]

                        for m in self.permuType:
                            permuMask = permuMask * (inputs['seman_gt'] != m)
                        permuMask = 1 - permuMask
                        permuMask = permuMask[:,:,1:-1,1:-1]

                    bdErr, rdErr = self.objReg.regularizeBuildingRoad(surnormMap, wallMask, roadMask)
                    padSize = int((self.varWindow-1) / 2)
                    permuMask = permuMask[:, :, padSize : -padSize, padSize : -padSize]
                    posVarErr = self.objReg.regularizePoleSign(surnormMap, permuMask)

                    loss = loss + (skyerr / 1000000 + bdErr / 20 + rdErr / 10 + posVarErr / 50) * self.opt.mulRegScale
                    if scale == 0:
                        if skyerr > 0:
                            losses["loss_reg/{}".format("sky")] = skyerr
                        if bdErr > 0:
                            losses["loss_reg/{}".format("building")] = bdErr
                        if rdErr > 0:
                            losses["loss_reg/{}".format("road")] = rdErr
                        if posVarErr > 0:
                            losses["loss_reg/{}".format("poleSign")] = posVarErr

                if self.opt.borderSim:
                    foreGroundMask = torch.ones(outputs[('disp', scale)].shape).cuda().byte()
                    with torch.no_grad():
                        for m in self.foregroundType:
                            foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                        foreGroundMask = 1 - foreGroundMask
                        foreGroundMask = foreGroundMask.float()
                    simBorderLoss, contrastBorderLoss = self.rdSampleOnBorder.randomSampleReg(outputs[('disp', scale)], foreGroundMask)
                    # loss += simBorderLoss * 0.1 + contrastBorderLoss * 0.001
                    # loss += simBorderLoss * 0.1 * self.opt.borderSimScale + contrastBorderLoss * 0.08 * self.opt.borderSimScale
                    # loss += simBorderLoss * 0.05 * self.opt.borderSimScale * self.opt.borderSmoothScale + contrastBorderLoss * 0.08 * self.opt.borderSimScale * self.opt.borderContrastScale
                    # loss += simBorderLoss * 0.7 * self.opt.borderSimScale + contrastBorderLoss * 0.2 * self.opt.borderSimScale
                    # loss += simBorderLoss * 0.01 * self.opt.borderSimScale * self.opt.borderSmoothScale + contrastBorderLoss * 0.2 * self.opt.borderSimScale
                    loss += simBorderLoss * 0.1 * self.opt.borderSimScale * self.opt.borderSmoothScale + contrastBorderLoss * 0.2 * self.opt.borderSimScale
                    if simBorderLoss != 0:
                        losses["loss_reg/{}".format("borderSimilar")] = simBorderLoss
                    if contrastBorderLoss != 0:
                        losses["loss_reg/{}".format("borderContrast")] = contrastBorderLoss
                    # self.rdSampleOnBorder

                if self.opt.borderRegression:
                    # foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]
                    # backgroundType = [0, 1, 2, 3, 4, 8, 9, 10]
                    # suppressType = [255]
                    foreGroundMask = torch.ones(outputs[('disp', scale)].shape).cuda().byte()
                    backGroundMask = torch.ones(outputs[('disp', scale)].shape).cuda().byte()
                    suppresMask = torch.ones(outputs[('disp', scale)].shape).cuda().byte()

                    with torch.no_grad():
                        for m in self.foregroundType:
                            foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                        foreGroundMask = 1 - foreGroundMask
                        for m in self.backgroundType:
                            backGroundMask = backGroundMask * (inputs['seman_gt'] != m)
                        backGroundMask = 1 - backGroundMask
                        for m in self.suppressType:
                            suppresMask = suppresMask * (inputs['seman_gt'] != m)
                        suppresMask = 1 - suppresMask
                        suppresMask = suppresMask.float()
                        combinedMask = torch.cat([foreGroundMask, backGroundMask], dim=1).float()

                    # borderRegFig = self.borderRegress.visualize_computeBorder(outputs[('disp', scale)], combinedMask, suppresMask = suppresMask, viewIndex=0)
                    # borderLoss1, borderLoss2 = self.borderRegress.borderRegression(outputs[('disp', scale)], combinedMask, suppresMask = suppresMask)
                    # loss += borderLoss1 * 0.1 * self.opt.borderRegScale + borderLoss2 * 0.1 * self.opt.borderRegScale
                    # if scale == 0:
                    #     losses["loss_reg/{}".format("borderReg")] = borderLoss1
                    #     losses["loss_reg/{}".format("borderReg_overflow")] = borderLoss2
                        # print("borderLoss: %f" % borderLoss1)
                    borderLoss = self.borderRegress.borderRegression(outputs[('disp', scale)], combinedMask, suppresMask=suppresMask, ssimMask = sourceSSIMMask)
                    loss += borderLoss * 0.1 * self.opt.borderRegScale
                    if scale == 0:
                        losses["loss_reg/{}".format("borderReg")] = borderLoss
                    # check
                    # viewInd = 0
                    # skyerrFig = self.objReg.visualize_regularizeSky(outputs[('depth', 0, scale)], skyMask, viewInd=viewInd)
                    # BdErrFig, viewRdErrFig = self.objReg.visualize_regularizeBuildingRoad(surnormMap, wallMask, roadMask, outputs[('disp', 0)], viewInd=viewInd)
                    # surVarFig = self.objReg.visualize_regularizePoleSign(surnormMap, permuMask, outputs[('disp', 0)], viewInd=viewInd)

                if self.opt.disparity_smoothness > 1e-10:
                    # if outputs[('mul_disp', scale)].shape[1] > 1:
                    #     mult_disp = outputs[('mul_disp', scale)][:,0:-1:,:]
                    # else:
                    #     mult_disp = outputs[('mul_disp', scale)]
                    mult_disp = outputs[('disp', scale)]
                    mean_disp = mult_disp.mean(2, True).mean(3, True)
                    norm_disp = mult_disp / (mean_disp + 1e-7)
                    # smooth_loss = get_smooth_loss(norm_disp, color)
                    smooth_loss = get_smooth_loss(norm_disp, target)
                    if scale == 0:
                        losses["loss_reg/{}".format("smooth")] = smooth_loss
                    loss = loss + self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            loss = loss / self.num_scales
            losses["loss_depth"] = ssimLossMean / self.num_scales
        if self.is_regress_semanticLoss(inputs, outputs):
            loss_seman, loss_semantoshow = self.semanticLoss(inputs, outputs, use_sep_semant_train = self.is_use_sep_seman_train()) # semantic loss is scaled already
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
        if not self.is_use_sep_seman_train():
            gt = inputs['seman_gt_eval'].cpu().numpy().astype(np.uint8)
        else:
            gt = inputs['seperate_seman_gt'][:,0,:,:].cpu().numpy().astype(np.uint8)
        pred = self.sfx(outputs[('seman', 0)]).detach()
        pred = torch.argmax(pred, dim=1).type(torch.float).unsqueeze(1)
        pred = F.interpolate(pred, [gt.shape[1], gt.shape[2]], mode='nearest')
        pred = pred.squeeze(1).cpu().numpy().astype(np.uint8)
        # visualize_semantic(gt[0,:,:]).show()
        # visualize_semantic(pred[0,:,:]).show()

        confMatrix = generateMatrix(args)
        # instStats = generateInstanceStats(args)
        # perImageStats = {}
        # nbPixels = 0

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



    def log(self, mode, inputs, outputs, losses, writeImage = False):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            if l != 'totLoss':
                writer.add_scalar("{}".format(l), v, self.step)

        if writeImage:
            cm = plt.get_cmap('magma')
            dispimg = outputs[("disp", 0)][0,0,:,:].cpu().numpy()
            dispimg = dispimg / 0.1
            viewmask = (cm(dispimg) * 255).astype(np.uint8)
            # pil.fromarray(viewmask).save("/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/trianReCompare/" + str(self.step) + ".png")

            slice = inputs['seman_gt'][0, 0, :, :].cpu().numpy()
            seman = visualize_semantic(slice)
            overlay = (np.array(viewmask[:,:,0:3]) * 0.7 + 0.3 * np.array(seman)).astype(np.uint8)

            dispSuppressed = (outputs[("disp", 0)] * (1 - outputs['ssimMask']))[0,0,:,:].cpu().numpy()
            dispSuppressed = dispSuppressed / 0.1
            viewSupmask = (cm(dispSuppressed) * 255).astype(np.uint8)
            supoverlay = (np.array(viewSupmask[:, :, 0:3]) * 0.7 + 0.3 * np.array(seman)).astype(np.uint8)
            overlay = np.concatenate([overlay, viewmask[:, :, 0:3], viewSupmask[:, :, 0:3], supoverlay[:,:,0:3]], axis=0)

            pil.fromarray(overlay).save("/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/trianReCompare/" + str(self.step) + ".png")
            # for j in range(min(4, self.opt.batch_size)):
            #     for s in self.opt.scales:
            #         for frame_id in self.opt.frame_ids:
            #             writer.add_image(
            #                 "color_{}_{}/{}".format(frame_id, s, j),
            #                 inputs[("color", frame_id, s)][j].data, self.step)
            #             if s == 0 and frame_id != 0:
            #                 writer.add_image(
            #                     "color_pred_{}_{}/{}".format(frame_id, s, j),
            #                     outputs[("color", frame_id, s)][j].data, self.step)
            #
            #         writer.add_image(
            #             "disp_{}/{}".format(s, j),
            #             normalize_image(outputs[("disp", s)][j]), self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

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

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
