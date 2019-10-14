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

splits_dir = os.path.join(os.path.dirname(__file__), "splits")
STEREO_SCALE_FACTOR = 5.4
class Tensor23dPts:
    def __init__(self, height = 375, width = 1242):
        self.height = height
        self.width = width
        xx, yy = np.meshgrid(np.arange(self.width), np.arange(self.height))
        self.xx = xx.flatten()
        self.yy = yy.flatten()
        objType = 19
        self.colorMap = np.zeros((objType + 1, self.xx.shape[0], 3), dtype=np.uint8)
        for i in range(objType):
            if i == objType:
                k = 255
            else:
                k = i
            self.colorMap[i, :, :] = np.repeat(np.expand_dims(np.array(trainId2label[k].color), 0), self.xx.shape[0], 0)
        self.colorMap = self.colorMap.astype(np.float)
        self.colorMap = self.colorMap / 255


    def visualize3d(self, depth, ind, intrinsic_in, extrinsic_in, gtmask_in = None, gtdepth_in = None, semanticMap = None, velo_in = None, rgb_in = None, disp_in = None):
        depth = F.interpolate(depth, size=[self.height, self.width], mode='bilinear', align_corners=False)
        gtdepth_in = F.interpolate(gtdepth_in, size=[self.height, self.width], mode='bilinear', align_corners=False)
        gtmask_in = F.interpolate(gtmask_in, size=[self.height, self.width], mode='nearest')
        intrinsic = intrinsic_in[ind, :, :].cpu().numpy()
        extrinsic = extrinsic_in[ind, :, :].cpu().numpy()
        gtmask = gtmask_in[ind, 0, :, :].cpu().numpy()
        gtdepth = gtdepth_in[ind, 0, :, :].cpu().numpy()
        velo = velo_in[ind, :, :].cpu().numpy()
        slice = depth[ind, 0, :, :].cpu().numpy()
        rgb = rgb_in[ind, :, :, :].permute(1,2,0).cpu().numpy()
        disp = disp_in[ind, 0, :, :].cpu().numpy()
        cm = plt.get_cmap('magma')
        assert depth.shape[1] == 1, "please input single channel depth map"

        im_shape = np.array([375, 1242])
        projectedPts2d = (intrinsic @ extrinsic @ velo.T).T
        projectedPts2d[:, 0] = projectedPts2d[:, 0] / projectedPts2d[:,2]
        projectedPts2d[:, 1] = projectedPts2d[:, 1] / projectedPts2d[:, 2]
        val_inds = (projectedPts2d[:, 0] >= 0) & (projectedPts2d[:, 1] >= 0)
        val_inds = val_inds & (projectedPts2d[:, 0] < im_shape[1]) & (projectedPts2d[:, 1] < im_shape[0])
        projectedPts2d = projectedPts2d[val_inds, :]
        velo = velo[val_inds,:]


        depthFlat = slice.flatten()
        oneColumn = np.ones(self.height * self.width)
        pixelLoc = np.stack([self.xx * depthFlat, self.yy * depthFlat, depthFlat, oneColumn], axis=1)
        veh_coord = (np.linalg.inv(intrinsic @ extrinsic) @ pixelLoc.T).T

        if gtmask is not None and gtdepth is not None:
            mask = gtmask == 1
            mask = mask.flatten()
            veh_coord = veh_coord[mask, :]
        if semanticMap is not None:
            semanticMap = semanticMap.cpu().numpy()
            semanticMap = semanticMap.flatten()
            semanticMap[semanticMap == 255] = 19
            colors = self.colorMap[semanticMap, np.arange(self.xx.shape[0]), :]
            if mask is not None:
                colors = colors[mask, :]


        camPos = (np.linalg.inv(extrinsic) @ np.array([0,0,0,1]).T).T
        # if gtmask is not None and gtdepth is not None:
        viewGtMask = copy.deepcopy(gtmask)
        viewGtMask = (cm(viewGtMask) * 255).astype(np.uint8)
        viewGtDepth = gtdepth
        vmax = np.percentile(viewGtDepth, 99)
        viewGtDepth = (cm(viewGtDepth / vmax) * 255).astype(np.uint8)
        viewDisp = copy.deepcopy(disp)
        vmax = np.percentile(viewDisp, 95)
        viewDisp = (cm(viewDisp / vmax) * 255).astype(np.uint8)
        viewRgb = copy.deepcopy(rgb)
        viewRgb = (viewRgb * 255).astype(np.uint8)



        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=6., azim=170)
        ax.dist = 4
        sampleDense = 1
        ax.scatter(veh_coord[0::sampleDense, 0], veh_coord[0::sampleDense, 1], veh_coord[0::sampleDense, 2], s=0.2, c='g')
        ax.scatter(velo[0::sampleDense, 0], velo[0::sampleDense, 1], velo[0::sampleDense, 2], s=0.2, c='r')
        # ax.scatter(camPos[0], camPos[1], camPos[2], s=10, c='g')
        ax.set_zlim(-10, 10)
        plt.ylim([-20, 20])
        plt.xlim([10, 16])
        set_axes_equal(ax)
        plt.close()


        tmpImgName = 'tmp1.png'
        fig.savefig(tmpImgName)
        plt.close(fig)
        img1 = pil.open(tmpImgName)
        return img1, veh_coord, velo


class ViewErrorMap():
    def __init__(self):
        self.idt = "errMap"
        self.cm = plt.get_cmap('spring')
    def viewErr(self, est, gt, viewInd = 0):
        width = gt.shape[3]
        height = gt.shape[2]
        est_resized = F.interpolate(est, [height, width], mode='bilinear', align_corners=False)
        gtMap = gt[viewInd, 0, :, :].cpu().numpy() + 1e-5
        estMap = est_resized[viewInd, 0, :, :].cpu().numpy()
        mask = gtMap > 1e-4
        thresh = np.maximum((gtMap / estMap), (estMap / gtMap)) * mask.astype(np.float)
        thresh = (thresh > 1.25).astype(np.float)

        viewdisp = (self.cm(thresh)* 255).astype(np.uint8)
        visRe = np.array(pil.fromarray(viewdisp))[:,:,0:3]
        visRe[np.logical_not(mask), :] = np.array([0,0,0])
        return pil.fromarray(visRe).resize([est.shape[3], est.shape[2]], resample=pil.BILINEAR), thresh, mask

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
    # encoder's record of height and weight are of less important now

    if opt.use_stereo:
        opt.frame_ids.append("s")
    if opt.dataset == 'cityscape':
        dataset = datasets.CITYSCAPERawDataset(opt.data_path, filenames,
                                           opt.height, opt.width, opt.frame_ids, 4, is_train=False, tag=opt.dataset, load_meta=True)
    elif opt.dataset == 'kitti':
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           opt.height, opt.width, opt.frame_ids, 4, is_train=False, tag=opt.dataset, is_load_semantics=True)
    else:
        raise ValueError("No predefined dataset")
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
                            pin_memory=True, drop_last=True)

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




    ##--------------------Visualization parameter here----------------------------##
    sfx = torch.nn.Softmax(dim=1)
    mergeDisp = Merge_MultDisp(opt.scales, batchSize = opt.batch_size, isMulChannel = opt.isMulChannel)
    svRoot = '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/figure_visual'
    index = 0
    isvisualize = True
    useGtSeman = True
    useSeman = False
    viewSurfaceNormal = False
    viewSelfOcclu = False
    viewMutuallyRegularizedBorder= False
    viewLiuSemanCompare = False
    viewSecondOrder = False
    viewBorderConverge = True
    expBin = True
    height = 288
    width = 960
    tensor23dPts = Tensor23dPts(height=height, width=width)

    dirpath = os.path.join(svRoot, opt.model_name)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if viewSurfaceNormal:
        compsn = ComputeSurfaceNormal(height = height, width = width, batch_size = opt.batch_size).cuda()

    if viewSelfOcclu:
        selfclu = SelfOccluMask().cuda()

    if viewMutuallyRegularizedBorder:
        mrb = MutuallyRegularizedBorders(height=height, width=width, batchsize=opt.batch_size)
        iouFore_gtdepth2gtseman = list()
        iouBack_gtdepth2gtseman = list()
        iouValid_gtdepth2gtseman = list()

        iouFore_estdepth2gtseman = list()
        iouBack_estdepth2gtseman = list()
        iouValid_estdepth2gtseman = list()

        iouFore_estdepth2estseman = list()
        iouBack_estdepth2estseman = list()
        iouValid_estdepth2estseman = list()

    if viewLiuSemanCompare:
        cmpBCons = computeBorderDistance()
        compGrad = computeGradient()
        semanest2semangt = np.zeros(31)
        depth2disp = np.zeros(31)
        depth2semangt = np.zeros(31)
        disp2semanest = np.zeros(31)
        sfx = torch.nn.Softmax(dim=1)
        cmpBCons.cuda()
        compGrad.cuda()

    if viewSecondOrder:
        compSecGrad = SecondOrderGrad().cuda()

    if viewBorderConverge:
        borderConverge = BorderConverge(height, width, opt.batch_size).cuda()

    if expBin:
        expbinmap = expBinaryMap(height, width, opt.batch_size).cuda()

    computedNum = 0
    # with torch.no_grad():
    for idx, inputs in enumerate(dataloader):
        for key, ipt in inputs.items():
            if not(key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta'):
                inputs[key] = ipt.to(torch.device("cuda"))
        input_color = inputs[("color", 0, 0)].cuda()
        features = encoder(input_color)
        outputs = dict()
        outputs.update(depth_decoder(features, computeSemantic=True, computeDepth=False))
        outputs.update(depth_decoder(features, computeSemantic=False, computeDepth=True))

        if isvisualize:
            if useGtSeman:
                mergeDisp(inputs, outputs, eval=False)
            else:
                mergeDisp(inputs, outputs, eval=True)

            dispMap = outputs[('disp', 0)]
            scaled_disp, depthMap = disp_to_depth(dispMap, 0.1, 100)
            depthMap = depthMap * STEREO_SCALE_FACTOR
            depthMap = torch.clamp(depthMap, max=80)

            if useGtSeman:
                fig_seman = tensor2semantic(inputs['seman_gt'], ind=index, isGt=True)
            else:
                if useSeman:
                    fig_seman = tensor2semantic(outputs[('seman', 0)], ind=index)
                else:
                    fig_seman = inputs[('color', 0, 0)][index, :, :, :].permute(1,2,0).cpu().numpy()
                    fig_seman = (fig_seman * 255).astype(np.uint8)
                    fig_seman = pil.fromarray(fig_seman)

            fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=index)
            fig_disp = tensor2disp(outputs[('disp', 0)], ind=index, vmax=0.1)

            gtmask = (inputs['depth_gt'] > 0).float()
            gtdepth = inputs['depth_gt']
            velo = inputs['velo']
            fig_3d, veh_coord, veh_coord_gt = tensor23dPts.visualize3d(depthMap.detach(), ind=index,
                                                                       intrinsic_in=inputs['realIn'],
                                                                       extrinsic_in=inputs['realEx'],
                                                                       gtmask_in=gtmask,
                                                                       gtdepth_in=gtdepth,
                                                                       semanticMap=None,
                                                                       velo_in=velo,
                                                                       rgb_in = inputs[('color', 's', 0)],
                                                                       disp_in = outputs[('disp', 0)].detach()
                                                                       )
            if viewMutuallyRegularizedBorder:
                foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18] # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
                backgroundType = [2, 3, 4, 8, 9, 10] #building, wall, fence, vegetation, terrain, sky
                foreGroundMask = torch.ones(dispMap.shape).cuda().byte()
                backGroundMask = torch.ones(dispMap.shape).cuda().byte()

                with torch.no_grad():
                    for m in foregroundType:
                        foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                    foreGroundMask = 1 - foreGroundMask
                    for m in backgroundType:
                        backGroundMask = backGroundMask * (inputs['seman_gt'] != m)
                    backGroundMask = 1 - backGroundMask

                # tensor2disp(foreGroundMask, ind=0, vmax=1).show()
                # tensor2disp(backGroundMask, ind=0, vmax=1).show()
                # tensor2rgb(inputs[('color', 0, 0)], ind=0).show()
                # tensor2semantic(inputs['seman_gt'],ind=0,isGt=True).show()
                iouForeMean, iouBackMean, isvalid = mrb.visualization(gtdepth, foreGroundMask, backGroundMask, viewind= index, rgb=inputs[('color', 0, 0)])
                iouFore_gtdepth2gtseman.append(iouForeMean)
                iouBack_gtdepth2gtseman.append(iouBackMean)
                iouValid_gtdepth2gtseman.append(isvalid)


                iouForeMean, iouBackMean, isvalid = mrb.visualization(1 - dispMap, foreGroundMask, backGroundMask,
                                                                      viewind=index, rgb=inputs[('color', 0, 0)])

                iouFore_estdepth2gtseman.append(iouForeMean)
                iouBack_estdepth2gtseman.append(iouBackMean)
                iouValid_estdepth2gtseman.append(isvalid)

                semanMapEst = outputs[('seman', 0)]
                semanMapEst_sfxed = sfx(semanMapEst)
                foreGroundMask_est = torch.sum(semanMapEst_sfxed[:, foregroundType, :, :], dim=1).unsqueeze(1)
                backGroundMask_est = torch.sum(semanMapEst_sfxed[:, backgroundType, :, :], dim=1).unsqueeze(1)
                other_est = 1 - (foreGroundMask_est + backGroundMask_est)
                tot_est = torch.cat([foreGroundMask_est, backGroundMask_est, other_est], dim=1)
                foreGroundMask_est_bin = (torch.argmax(tot_est, dim=1) == 0).unsqueeze(1)
                backGroundMask_est_bin = (torch.argmax(tot_est, dim=1) == 1).unsqueeze(1)
                iouForeMean, iouBackMean, isvalid = mrb.visualization(1 - dispMap, foreGroundMask_est_bin, backGroundMask_est_bin,
                                                                      viewind=index, rgb=inputs[('color', 0, 0)])
                iouFore_estdepth2estseman.append(iouForeMean)
                iouBack_estdepth2estseman.append(iouBackMean)
                iouValid_estdepth2estseman.append(isvalid)

                # tensor2disp(foreGroundMask_est_bin, vmax=1, ind=0).show()
                # tensor2disp(backGroundMask_est_bin, vmax=1, ind=0).show()
            if viewLiuSemanCompare:
                foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18] # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
                backgroundType = [2, 3, 4, 8, 9, 10] #building, wall, fence, vegetation, terrain, sky
                foreGroundMask = torch.ones(dispMap.shape).cuda().byte()
                backGroundMask = torch.ones(dispMap.shape).cuda().byte()

                with torch.no_grad():
                    for m in foregroundType:
                        foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                    foreGroundMask = 1 - foreGroundMask
                    for m in backgroundType:
                        backGroundMask = backGroundMask * (inputs['seman_gt'] != m)
                    backGroundMask = 1 - backGroundMask

                dispMapEst = outputs[('disp', 0)]
                semanMapEst = outputs[('seman', 0)]
                semanMapGt = inputs['seman_gt']
                depthMapGt = inputs['depth_gt']

                sparseDepthmapGrad = compGrad.computegrad11_sparse(depthMapGt)
                sparseDepthmapGrad_bin = sparseDepthmapGrad > 0
                sparseDepthmapGrad = F.interpolate(sparseDepthmapGrad, [height, width], mode='bilinear', align_corners=True)
                sparseDepthmapGrad_bin = F.interpolate(sparseDepthmapGrad_bin.float(), [height, width], mode='nearest')
                sparseDepthmapGrad = sparseDepthmapGrad * sparseDepthmapGrad_bin
                # depthMapGt_bin = depthMapGt > 1e-1
                # depthMapGt = F.interpolate(sparseDepthmapGrad, (height, width), mode='bilinear', align_corners=False)
                # depthMapGt_bin = F.interpolate(depthMapGt_bin.float(), (height, width), mode='nearest')
                # depthMapGt = depthMapGt * depthMapGt_bin
                # compGrad.computegrad11_sparse(depthMapGt)
                # tensor2disp(depthMapGt>0, ind=0, vmax=1).show()


                semanMapEst_sfxed = sfx(semanMapEst)
                semanMapEst_inds = torch.argmax(semanMapEst_sfxed, dim=1).unsqueeze(1)
                seman_est_fig = tensor2semantic(semanMapEst_inds, ind=0)
                seman_gt_fig = tensor2semantic(semanMapGt, ind=0)
                depthMapGt_fig = tensor2disp(depthMapGt, ind=0, vmax=20)
                depthMapGt_fig = depthMapGt_fig.resize((width, height), resample=pil.BILINEAR)


                foreGroundMask_est = torch.sum(semanMapEst_sfxed[:,foregroundType,:,:], dim=1).unsqueeze(1)

                dispMapGrad = compGrad.computegrad11(dispMapEst)
                foreGroundMaskGrad = compGrad.computegrad11(foreGroundMask.float())
                foreGroundMask_estGrad = compGrad.computegrad11(foreGroundMask_est)
                sparseDepthmapGrad_fig = tensor2disp(sparseDepthmapGrad, ind=0, vmax=20)
                dispMapGrad_fig = tensor2disp(dispMapGrad, ind=0, vmax=0.08)
                foreGroundMaskGrad_fig = tensor2disp(foreGroundMaskGrad, ind=0, vmax=1)
                foreGroundMask_estGrad_fig = tensor2disp(foreGroundMask_estGrad, ind=0, vmax=1.5)

                dispMapGrad_bin = dispMapGrad > 0.011
                foreGroundMaskGrad_bin = foreGroundMaskGrad > 0.5
                foreGroundMask_estGrad_bin = foreGroundMask_estGrad > 0.6
                sparseDepthmapGrad_bin = sparseDepthmapGrad > 9
                dispMapGrad_bin_fig = tensor2disp(dispMapGrad_bin, ind=0, vmax=1)
                foreGroundMaskGrad_bin_fig = tensor2disp(foreGroundMaskGrad_bin, ind=0, vmax=1)
                foreGroundMask_estGrad_bin_fig = tensor2disp(foreGroundMask_estGrad_bin, ind=0, vmax=1)
                sparseDepthmapGrad_bin_fig = tensor2disp(sparseDepthmapGrad_bin, ind=0, vmax=1)

                visualizeImage = np.concatenate([np.array(fig_rgb), np.array(fig_disp)[:,:,0:3], np.array(seman_est_fig), np.array(seman_gt_fig), np.array(depthMapGt_fig)[:,:,0:3]], axis=0)
                visualizeImage_grad = np.concatenate([np.array(fig_rgb), np.array(dispMapGrad_fig)[:,:,0:3], np.array(foreGroundMask_estGrad_fig)[:,:,0:3], np.array(foreGroundMaskGrad_fig)[:,:,0:3], np.array(sparseDepthmapGrad_fig)[:,:,0:3]], axis=0)
                visualizeimage_grad_bin = np.concatenate([np.array(fig_rgb), np.array(dispMapGrad_bin_fig)[:,:,0:3], np.array(foreGroundMask_estGrad_bin_fig)[:,:,0:3], np.array(foreGroundMaskGrad_bin_fig)[:,:,0:3], np.array(sparseDepthmapGrad_bin_fig)[:,:,0:3]], axis=0)
                tot = np.concatenate([np.array(visualizeImage), np.array(visualizeImage_grad), np.array(visualizeimage_grad_bin)], axis=1)
                pil.fromarray(tot).save('/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderConsistAnalysis/%d.png' % idx)
                # pil.fromarray(tot).show()
                # pil.fromarray(visualizeImage).show()
                # pil.fromarray(visualizeImage_grad).show()
                # pil.fromarray(visualizeimage_grad_bin).show()


                semanest2semangt = semanest2semangt + cmpBCons.computeDistance(foreGroundMask_estGrad_bin, foreGroundMaskGrad_bin)
                depth2disp = depth2disp + cmpBCons.computeDistance(sparseDepthmapGrad_bin, dispMapGrad_bin)
                depth2semangt = depth2semangt + cmpBCons.computeDistance(sparseDepthmapGrad_bin, foreGroundMaskGrad_bin)
                disp2semanest = disp2semanest + cmpBCons.computeDistance(dispMapGrad_bin, foreGroundMask_estGrad_bin)

                # tensor2disp(dispMapEst, ind=index, percentile=90).show()

            if viewBorderConverge:
                semanMapEst = outputs[('seman', 0)]
                semanMapEst_sfxed = sfx(semanMapEst)
                foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17,
                                  18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
                foreGroundMask_est = torch.sum(semanMapEst_sfxed[:, foregroundType, :, :], dim=1).unsqueeze(1)
                dispMapEst = outputs[('disp', 0)]

                # borderConverge.visualization(dispMapEst, foreGroundMask_est)
                if expBin:
                    expbinmap.visualization3(disparity=dispMapEst, semantics=foreGroundMask_est)
                a = 1

            if viewSecondOrder:
                disp2order = compSecGrad.computegrad11(outputs[('disp', 0)])
                tensor2disp(disp2order, ind=0, percentile=95).show()

            if viewSurfaceNormal:
                surnorm = compsn.visualize(depthMap=depthMap, invcamK=inputs['invcamK'].cuda().float(), orgEstPts=veh_coord,
                                           gtEstPts=veh_coord_gt, viewindex=index)
                surnormMap = compsn(depthMap=depthMap, invcamK=inputs['invcamK'].cuda().float())

            if viewSelfOcclu:
                fl = inputs[("K", 0)][:, 0, 0]
                bs = torch.abs(inputs["stereo_T"][:, 0, 3])
                clufig, suppressedDisp = selfclu.visualize(dispMap, viewind=index)

            if viewSurfaceNormal and viewSelfOcclu:
                surnorm = surnorm.resize([width, height])
                surnorm_mixed = pil.fromarray(
                    (np.array(surnorm) * 0.2 + np.array(fig_disp)[:, :, 0:3] * 0.8).astype(np.uint8))
                disp_seman = (np.array(fig_disp)[:, :, 0:3].astype(np.float) * 0.8 + np.array(fig_seman).astype(
                    np.float) * 0.2).astype(np.uint8)
                supprressed_disp_seman = (np.array(suppressedDisp)[:, :, 0:3].astype(np.float) * 0.8 + np.array(fig_seman).astype(
                    np.float) * 0.2).astype(np.uint8)
                rgb_seman = (np.array(fig_seman).astype(np.float) * 0.5 + np.array(fig_rgb).astype(
                    np.float) * 0.5).astype(np.uint8)

                # clud_disp = (np.array(clufig)[:, :, 0:3].astype(np.float) * 0.3 + np.array(fig_disp)[:, :, 0:3].astype(
                #     np.float) * 0.7).astype(np.uint8)
                comb1 = np.concatenate([np.array(supprressed_disp_seman)[:, :, 0:3], np.array(suppressedDisp)[:, :, 0:3]], axis=1)
                comb2 = np.concatenate([np.array(disp_seman)[:, :, 0:3], np.array(fig_disp)[:, :, 0:3]], axis=1)
                # comb3 = np.concatenate([np.array(errFig)[:, :, 0:3], np.array(surnorm)[:, :, 0:3]], axis=1)
                comb4 = np.concatenate([np.array(fig_seman)[:, :, 0:3], np.array(rgb_seman)[:, :, 0:3]],
                                       axis=1)
                comb6 = np.concatenate([np.array(clufig)[:, :, 0:3], np.array(fig_disp)[:, :, 0:3]], axis=1)

                fig3dsize = np.ceil(np.array([comb4.shape[1] , comb4.shape[1] / fig_3d.size[0] * fig_3d.size[1]])).astype(np.int)
                comb5 = np.array(fig_3d.resize(fig3dsize))

            # fig = pil.fromarray(combined)
            # fig.save(os.path.join(dirpath, str(idx) + '.png'))
            print("%dth img finished" % idx)
            # if idx >=4:
            #     break
    if viewLiuSemanCompare:
        semanest2semangt_p = semanest2semangt / np.sum(semanest2semangt)
        semanest2semangt_p_ = semanest2semangt_p[0:-1]
        mean = np.sum(np.arange(len(semanest2semangt_p_)) * semanest2semangt_p_)
        std = np.sqrt(np.sum((np.arange(len(semanest2semangt_p_)) - mean) ** 2 * semanest2semangt_p_))
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(semanest2semangt_p)), semanest2semangt_p)
        ax.set_ylabel('Percentile')
        ax.set_xlabel('Distance in pixel, mean %f, std %f' % (mean, std))
        ax.set_title("Pixel distance of semantic, est to gt")
        fig.savefig("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderConsistAnalysis/seman_est2gt.png")
        plt.close(fig)

        depth2disp_p = depth2disp / np.sum(depth2disp)
        depth2disp_p_ = depth2disp_p[0:-1]
        mean = np.sum(np.arange(len(depth2disp_p_)) * depth2disp_p_)
        std = np.sqrt(np.sum((np.arange(len(depth2disp_p_)) - mean) ** 2 * depth2disp_p_))
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(depth2disp_p)), depth2disp_p)
        ax.set_ylabel('Percentile')
        ax.set_xlabel('Distance in pixel, mean %f, std %f' % (mean, std))
        ax.set_title("Pixel distance of depth, gt to est")
        fig.savefig("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderConsistAnalysis/depth_gt2est.png")
        plt.close(fig)

        depth2semangt_p = depth2semangt / np.sum(depth2semangt)
        depth2semangt_p_ = depth2semangt_p[0:-1]
        mean = np.sum(np.arange(len(depth2semangt_p_)) * depth2semangt_p_)
        std = np.sqrt(np.sum((np.arange(len(depth2semangt_p_)) - mean) ** 2 * depth2semangt_p_))
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(depth2semangt_p)), depth2semangt_p)
        ax.set_ylabel('Percentile')
        ax.set_xlabel('Distance in pixel, mean %f, std %f' % (mean, std))
        ax.set_title("Pixel distance of depth and semantic, gt")
        fig.savefig("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderConsistAnalysis/depth2seman_gt.png")
        plt.close(fig)

        disp2semanest_p = disp2semanest / np.sum(disp2semanest)
        disp2semanest_p_ = disp2semanest_p[0:-1]
        mean = np.sum(np.arange(len(disp2semanest_p_)) * disp2semanest_p_)
        std = np.sqrt(np.sum((np.arange(len(disp2semanest_p_)) - mean) ** 2 * disp2semanest_p_))
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(disp2semanest_p)), disp2semanest_p)
        ax.set_ylabel('Percentile')
        ax.set_xlabel('Distance in pixel, mean %f, std %f' % (mean, std))
        ax.set_title("Pixel distance of depth and semantic, est")
        fig.savefig("/media/shengjie/other/sceneUnderstanding/SDNET/visualization/borderConsistAnalysis/depth2seman_est.png")
        plt.close(fig)

    if viewMutuallyRegularizedBorder:
        iouFore_gtdepth2gtseman = np.array(iouFore_gtdepth2gtseman)
        iouBack_gtdepth2gtseman = np.array(iouBack_gtdepth2gtseman)
        iouValid_gtdepth2gtseman = np.array(iouValid_gtdepth2gtseman)
        iouFore_gtdepth2gtsemanMean = np.sum(iouFore_gtdepth2gtseman * iouValid_gtdepth2gtseman) / np.sum(iouValid_gtdepth2gtseman)
        iouBack_gtdepth2gtsemanMean = np.sum(iouBack_gtdepth2gtseman * iouValid_gtdepth2gtseman) / np.sum(iouValid_gtdepth2gtseman)

        iouFore_estdepth2gtseman = np.array(iouFore_estdepth2gtseman)
        iouBack_estdepth2gtseman = np.array(iouBack_estdepth2gtseman)
        iouValid_estdepth2gtseman = np.array(iouValid_estdepth2gtseman)
        iouFore_estdepth2gtsemanMean = np.sum(iouFore_estdepth2gtseman * iouValid_estdepth2gtseman) / np.sum(iouValid_estdepth2gtseman)
        iouBack_estdepth2gtsemanMean = np.sum(iouBack_estdepth2gtseman * iouValid_estdepth2gtseman) / np.sum(iouValid_estdepth2gtseman)

        iouFore_estdepth2estseman = np.array(iouFore_estdepth2estseman)
        iouBack_estdepth2estseman = np.array(iouBack_estdepth2estseman)
        iouValid_estdepth2estseman = np.array(iouValid_estdepth2estseman)
        iouFore_estdepth2estsemanMean = np.sum(iouFore_estdepth2estseman * iouValid_estdepth2estseman) / np.sum(iouValid_estdepth2estseman)
        iouBack_estdepth2estsemanMean = np.sum(iouBack_estdepth2estseman * iouValid_estdepth2estseman) / np.sum(iouValid_estdepth2estseman)

        print("iouFore_gtdepth2gtsemanMean is % f" % iouFore_gtdepth2gtsemanMean)
        print("iouBack_gtdepth2gtsemanMean is % f" % iouBack_gtdepth2gtsemanMean)
        print("iouFore_estdepth2gtsemanMean is % f" % iouFore_estdepth2gtsemanMean)
        print("iouBack_estdepth2gtsemanMean is % f" % iouBack_estdepth2gtsemanMean)
        print("iouFore_estdepth2estsemanMean is % f" % iouFore_estdepth2estsemanMean)
        print("iouBack_estdepth2estsemanMean is % f" % iouBack_estdepth2estsemanMean)
if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
