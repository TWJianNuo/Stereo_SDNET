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

def tensor2rgb(tensor, ind):
    slice = (tensor[ind, :, :, :].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    # pil.fromarray(slice).show()
    return pil.fromarray(slice)

def tensor2semantic(tensor, ind, isGt = False):
    slice = tensor[ind, :, :, :]
    if not isGt:
        slice = F.softmax(slice, dim=0)
        slice = torch.argmax(slice, dim=0).cpu().numpy()
    else:
        slice = slice[0,:,:].cpu().numpy()
    # visualize_semantic(slice).show()
    return visualize_semantic(slice)

def tensor2disp(tensor, ind, vmax = None):
    # slice = tensor[]
    # plt.imsave(name_dest_im, disp_resized_np, cmap='magma', vmax=vmax)
    slice = tensor[ind, 0, :, :].cpu().numpy()
    if vmax is None:
        vmax = np.percentile(slice, 90)
    slice = slice / vmax
    # slice = slice / slice.max()
    cm = plt.get_cmap('magma')
    slice = (cm(slice) * 255).astype(np.uint8)
    # pil.fromarray(slice).show()
    return pil.fromarray(slice)

class Tensor23dPts:
    def __init__(self):
        self.height = 1024
        self.width = 2048
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


    def visualize3d(self, tensor, ind, intrinsic, extrinsic, gtmask = None, gtdepth = None, semanticMap = None):
        assert tensor.shape[1] == 1, "please input single channel depth map"
        self.height = 1024
        self.width = 2048
        tensor = F.interpolate(tensor, [self.height, self.width], mode="bilinear", align_corners=False)
        intrinsic = intrinsic.cpu().numpy()
        extrinsic = extrinsic.cpu().numpy()
        slice = tensor[ind, 0, :, :].cpu().numpy()

        depthFlat = slice.flatten()
        oneColumn = np.ones(self.height * self.width)
        pixelLoc = np.stack([self.xx * depthFlat, self.yy * depthFlat, depthFlat, oneColumn], axis=1)
        cam_coord = (np.linalg.inv(intrinsic) @ pixelLoc.T).T
        veh_coord = (np.linalg.inv(extrinsic) @ cam_coord.T).T
        colors = None

        if gtmask is not None and gtdepth is not None:
            gtdepth = gtdepth.cpu().numpy()
            mask = gtdepth > 0
            # mask = gtdepth > -1000000
            mask = mask.flatten()
            depthFlat = gtdepth.flatten()
            oneColumn = np.ones(gtdepth.shape[0] * gtdepth.shape[1])
            pixelLoc = np.stack([self.xx[mask] * depthFlat[mask], self.yy[mask] * depthFlat[mask], depthFlat[mask], oneColumn[mask]], axis=1)
            cam_coord = (np.linalg.inv(intrinsic) @ pixelLoc.T).T
            veh_coord_2 = (np.linalg.inv(extrinsic) @ cam_coord.T).T

            veh_coord = veh_coord[mask, :]
        if semanticMap is not None:
            semanticMap = semanticMap.cpu().numpy()
            semanticMap = semanticMap.flatten()
            semanticMap[semanticMap == 255] = 19
            colors = self.colorMap[semanticMap, np.arange(self.xx.shape[0]), :]
            if mask is not None:
                colors = colors[mask, :]


        camPos = (np.linalg.inv(extrinsic) @ np.array([0,0,0,1]).T).T
        veh_coord[:, 0:3] = veh_coord[:, 0:3] - np.repeat(np.expand_dims(camPos, 0)[:,0:3], veh_coord.shape[0], 0)
        veh_coord_2[:, 0:3] = veh_coord_2[:, 0:3] - np.repeat(np.expand_dims(camPos, 0)[:,0:3], veh_coord_2.shape[0], 0)

        tmpImgName = 'tmp1.png'
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=6., azim=170)
        ax.dist = 4
        if colors is None:
            ax.scatter(veh_coord[0::100,0], veh_coord[0::100,1], veh_coord[0::100,2], s=0.1, c = 'b')
            ax.scatter(veh_coord_2[0::100, 0], veh_coord_2[0::100, 1], veh_coord_2[0::100, 2], s=0.1, c='r')
        else:
            ax.scatter(veh_coord_2[0::50, 0], veh_coord_2[0::50, 1], veh_coord_2[0::50, 2], s=0.5, c = colors[0::50, :])
        ax.scatter(camPos[0], camPos[1], camPos[2], s=10, c='g')
        ax.set_zlim(-10, 10)
        plt.ylim([-10, 10])
        plt.xlim([10, 16])
        set_axes_equal(ax)
        fig.savefig(tmpImgName)
        plt.close(fig)
        img1 = pil.open(tmpImgName)

        tmpImgName = 'tmp2.png'
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.view_init(elev=6., azim=170)
        ax.dist = 4
        if colors is None:
            ax.scatter(veh_coord[0::100,0], veh_coord[0::100,1], veh_coord[0::100,2], s=0.1, c = 'b')
            ax.scatter(veh_coord_2[0::100, 0], veh_coord_2[0::100, 1], veh_coord_2[0::100, 2], s=0.1, c='r')
        else:
            ax.scatter(veh_coord[0::50,0], veh_coord[0::50,1], veh_coord[0::50,2], s=0.5, c = colors[0::50, :])
        ax.scatter(camPos[0], camPos[1], camPos[2], s=10, c='g')
        ax.set_zlim(-10, 10)
        plt.ylim([-10, 10])
        plt.xlim([10, 16])
        set_axes_equal(ax)
        fig.savefig(tmpImgName)
        plt.close(fig)
        img2 = pil.open(tmpImgName)
        img = Image.fromarray(np.concatenate([np.array(img1)[:,:,0:3], np.array(img2)[:,:,0:3]], axis=1))

        return img, veh_coord, veh_coord_2

class Comp1dgrad(nn.Module):
    def __init__(self):
        super(Comp1dgrad, self).__init__()
        self.act = nn.Sigmoid()
        self.gradth = 0.1
        self.init_gradconv()
        # self.init_gaussconv(kernel_size=3, sigma=2, channels=1)



    # def init_gaussconv(self, kernel_size=3, sigma=2, channels=1):
    #     self.gaussconv = get_gaussian_kernel(kernel_size=kernel_size, sigma=sigma, channels=channels)
    #     self.gaussconv.cuda()
    def init_gradconv(self):
        weightsx = torch.Tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([[1., 2., 1.],
                                [0., 0., 0.],
                                [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)
        self.convy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=0, bias=False)

        self.convx.weight = nn.Parameter(weightsx,requires_grad=False)
        self.convy.weight = nn.Parameter(weightsy,requires_grad=False)

        self.convx.cuda()
        self.convy.cuda()
        # self.gaussKernel =
    def forward(self, tensor, boot = 1):
        # tensor_blurred = self.gaussconv(tensor)
        tensor_blurred = tensor
        grad_x = torch.abs(self.convx(tensor_blurred))
        grad_y = torch.abs(self.convy(tensor_blurred))
        # grad = (grad_x + grad_y - 0.012) / tensor_blurred[:,:,1:-1,1:-1] * 200 - 6
        grad = (grad_x + grad_y - 0.012) / tensor_blurred[:, :, 1:-1, 1:-1] * 10 - 6
        grad = self.act(grad)
        # vmax = np.percentile(grad.cpu().numpy(), 99)
        # a = grad.cpu().numpy()

        # grad_x = torch.abs(tensor[:, :, :-1, :-1] - tensor[:, :, :-1, 1:])
        # grad_y = torch.abs(tensor[:, :, :-1, :-1] - tensor[:, :, 1:, :-1])
        # grad = (grad_x + grad_y) * boostParam
        # grad = self.act(grad)
        # tensor2disp(grad, ind = 0, vmax=1).show()
        # tensor2disp(tensor, ind = 0).show()
        return grad


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

    if opt.use_stereo:
        opt.frame_ids.append("s")
    if opt.dataset == 'cityscape':
        dataset = datasets.CITYSCAPERawDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'], opt.frame_ids, 4, is_train=False, tag=opt.dataset, load_meta=True, is_sep_train_seman = False)
    elif opt.dataset == 'kitti':
        dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                           encoder_dict['height'], encoder_dict['width'], opt.frame_ids, 4, is_train=False, tag=opt.dataset)
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

    # x = torch.ones(2, 2, requires_grad=True)
    # print(x)
    # y = x + 2 + x
    # y = y.detach()
    # print(y)
    # z = y * y * 3
    # out = z.mean()
    # print(z, out)
    # out.backward()
    # print(x.grad)

    ##--------------------Visualization parameter here----------------------------##
    sfx = torch.nn.Softmax(dim=1)
    mergeDisp = Merge_MultDisp(opt.scales, batchSize = opt.batch_size, isMulChannel = opt.isMulChannel)
    svRoot = '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/figure_visual'
    index = 0
    isvisualize = True
    viewEdgeMerge = False
    isHist = False
    useGtSeman = True
    viewSurfaceNormal = True
    viewSelfOcclu = True
    viewDispUp = True
    viewSmooth = True
    viewMulReg = True
    viewBorderRegress = False
    viewBorderSimilarity = False
    viewRandomSample = True
    viewSemanReg = False
    viewDepthGuess = False
    height = 256
    width = 512
    tensor23dPts = Tensor23dPts()

    if isHist:
        rec = np.zeros((19,100))

    if opt.isMulChannel:
        app = os.path.join('mulDispOn', opt.model_name)
    else:
        app = os.path.join('mulDispOff', opt.model_name)

    dirpath = os.path.join(svRoot, app)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if viewEdgeMerge:
        comp1dgrad = Comp1dgrad().cuda()

    if viewSurfaceNormal:
        compsn = ComputeSurfaceNormal(height = height, width = width, batch_size = opt.batch_size).cuda()

    if viewSelfOcclu:
        selfclu = SelfOccluMask().cuda()

    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            # if idx != 12:
            #     continue
            for key, ipt in inputs.items():
                if not(key == 'height' or key == 'width' or key == 'tag' or key == 'cts_meta'):
                    inputs[key] = ipt.to(torch.device("cuda"))
            input_color = inputs[("color", 0, 0)].cuda()
            # input_color = torch.flip(input_color, dims=[3])
            features = encoder(input_color)
            outputs = dict()
            outputs.update(depth_decoder(features, computeSemantic=True, computeDepth=False))
            outputs.update(depth_decoder(features, computeSemantic=False, computeDepth=True))

            # view the processed semantic seperate training data
            # for viewInd in range(opt.batch_size):
            #     label = inputs['semanTrain_label']
            #     visualize_semantic(label[viewInd, 0, :, :].cpu().numpy()).show()
            #     fig_rgb = inputs['semanTrain_rgb'][viewInd, :, :, :].permute(1, 2, 0).cpu().numpy()
            #     fig_rgb = (fig_rgb * 255).astype(np.uint8)
            #     fig_rgb = pil.fromarray(fig_rgb)
            #     fig_rgb.show()


            if isHist:
                mulDisp = outputs[('mul_disp', 0)]
                scaled_disp, mulDepth = disp_to_depth(mulDisp, 0.1, 100)
                mulDepth = mulDepth.cpu()
                for i in range(mulDisp.shape[1]):
                    rec[i,:] += torch.histc(mulDepth[:,i,:,:],bins=100,min=0,max=100).numpy()

            if isvisualize:
                if useGtSeman:
                    # outputs[('mul_disp', 0)][:,2,:,:] = outputs[('mul_disp', 0)][:,2,:,:] * 0
                    # outputs[('mul_disp', 0)][:, 12, :, :] = outputs[('mul_disp', 0)][:, 12, :, :] * 0
                    mergeDisp(inputs, outputs, eval=False)
                else:
                    mergeDisp(inputs, outputs, eval=True)

                dispMap = outputs[('disp', 0)]
                scaled_disp, depthMap = disp_to_depth(dispMap, 0.1, 100)
                depthMap = depthMap * STEREO_SCALE_FACTOR
                # _, mul_depthMap = disp_to_depth(outputs[('mul_disp', 0)], 0.1, 100)
                # mul_depthMap = mul_depthMap * STEREO_SCALE_FACTOR

                if viewDispUp:
                    fig_dispup = compDispUp.visualize(scaled_disp, viewindex=index)

                if viewSmooth:
                    rgb = inputs[('color_aug', 0, 0)]
                    smoothfig = comSmooth.visualize(rgb=rgb, disp=scaled_disp, viewindex=index)

                if useGtSeman:
                    fig_seman = tensor2semantic(inputs['seman_gt'], ind=index, isGt=True)
                else:
                    fig_seman = tensor2semantic(outputs[('seman', 0)], ind=index)

                if viewSemanReg:
                    foregroundType = [11, 12, 13, 14, 15, 16, 17, 18]  # person, rider, car, truck, bus, train, motorcycle, bicycle
                    softmaxedSeman = F.softmax(outputs[('seman', 0)], dim=1)
                    forePredMask = torch.sum(softmaxedSeman[:,foregroundType,:,:], dim=1, keepdim=True)
                    foreGtMask = torch.ones(dispMap.shape).cuda().byte()

                    for m in foregroundType:
                        foreGtMask = foreGtMask * (inputs['seman_gt'] != m)
                    foreGtMask = 1 - foreGtMask
                    foreGtMask = foreGtMask.float()

                    forePredMask[forePredMask > 0.5] = 1
                    forePredMask[forePredMask <= 0.5] = 0

                    forePredMask = foreGtMask
                    rdSampleSeman.visualizeBorderSample(dispMap, forePredMask, gtMask=foreGtMask, viewIndex=index)


                    cm = plt.get_cmap('magma')
                    viewForePred = forePredMask[index, :, :, :].squeeze(0).detach().cpu().numpy()
                    viewForePred = (cm(viewForePred) * 255).astype(np.uint8)
                    # pil.fromarray(viewForePred).show()

                    viewForeGt = foreGtMask[index, :, :, :].squeeze(0).detach().cpu().numpy()
                    viewForeGt = (cm(viewForeGt) * 255).astype(np.uint8)
                    # pil.fromarray(viewForeGt).show()
                    forePredictCombined = np.concatenate([viewForePred, viewForeGt], axis=0)
                    # pil.fromarray(forePredictCombined).show()
                    pil.fromarray(forePredictCombined).save(os.path.join(dirpath, str(idx) + '_fg.png'))

                if viewDepthGuess:
                    wallType = [2, 3, 4] # Building, wall, fence
                    roadType = [0, 1, 9] # road, sidewalk, terrain
                    foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle

                    wallTypeMask = torch.ones(dispMap.shape).cuda().byte()
                    roadTypeMask = torch.ones(dispMap.shape).cuda().byte()
                    foreGroundMask = torch.ones(dispMap.shape).cuda().byte()

                    with torch.no_grad():
                        for m in wallType:
                            wallTypeMask = wallTypeMask * (inputs['seman_gt'] != m)
                        wallTypeMask = (1 - wallTypeMask).float()

                        for m in roadType:
                            roadTypeMask = roadTypeMask * (inputs['seman_gt'] != m)
                        roadTypeMask = (1 - roadTypeMask).float()

                        for m in foregroundType:
                            foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                        foreGroundMask = (1 - foreGroundMask).float()
                    originalSieze = [2048, 1024]
                    # currentSize = np.array([dispMap.shape[3], dispMap.shape[2]])
                    # scaleFac = np.eye(4)
                    # scaleFac[0,0] = currentSize[0] / originalSieze[0]
                    # scaleFac[1,1] = currentSize[1] / originalSieze[1]
                    # scaleFac = torch.Tensor(scaleFac).view(1,4,4).repeat(opt.batch_size, 1, 1).cuda()
                    # scaledIntrinsic = scaleFac @ inputs['realIn']
                    scaledIntrinsic = inputs['realIn']
                    depthGuess.visualizeDepthGuess(realDepth=depthMap, dispAct=dispMap, foredgroundMask = foreGroundMask, wallTypeMask=wallTypeMask, groundTypeMask=roadTypeMask, intrinsic= scaledIntrinsic, extrinsic=inputs['realEx'], semantic = inputs['seman_gt_eval'], cts_meta = inputs['cts_meta'], viewInd=index)
                    # realDepth, foredgroundMask, wallTypeMask, groundTypeMask, intrinsic, extrinsic

                fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=index)
                fig_disp = tensor2disp(outputs[('disp', 0)], ind=index)
                fig_3d, veh_coord, veh_coord_gt = tensor23dPts.visualize3d(depthMap, ind = index, intrinsic= inputs['cts_meta']['intrinsic'][index, :, :], extrinsic= inputs['cts_meta']['extrinsic'][index, :, :], gtmask=inputs['cts_meta']['mask'][index, :, :], gtdepth=inputs['cts_meta']['depthMap'][index, :, :], semanticMap=inputs['seman_gt_eval'][index, :, :])
                # check:
                # torch.inverse(inputs['invcamK'][index, :, :] @ inputs['realIn'][index, :, :]) - inputs['cts_meta']['extrinsic'][index, :, :]
                fig_grad = None

                if viewSurfaceNormal:
                    # surnorm = compsn.visualize(depthMap = depthMap, invcamK = inputs['invcamK'].cuda(), orgEstPts = veh_coord, gtEstPts = veh_coord_gt, viewindex = index)
                    surnorm = compsn.visualize(depthMap=depthMap, invcamK=inputs['invcamK'].cuda(), orgEstPts=veh_coord,
                                               gtEstPts=veh_coord_gt, viewindex=index)
                    surnormMap = compsn(depthMap=depthMap, invcamK=inputs['invcamK'].cuda())

                if viewMulReg:
                    depthMapLoc = depthMap / STEREO_SCALE_FACTOR
                    skyId = 10
                    skyMask = inputs['seman_gt'] == skyId
                    skyerr = objReg.visualize_regularizeSky(depthMapLoc, skyMask, viewInd=index)


                    wallType = [2, 3, 4] # Building, wall, fence
                    roadType = [0, 1, 9] # road, sidewalk, terrain
                    permuType = [5, 7] # Pole, traffic sign
                    chanWinSize = 5

                    wallMask = torch.ones_like(skyMask)
                    roadMask = torch.ones_like(skyMask)
                    permuMask = torch.ones_like(skyMask)

                    with torch.no_grad():
                        for m in wallType:
                            wallMask = wallMask * (inputs['seman_gt'] != m)
                        wallMask = 1 - wallMask
                        wallMask = wallMask[:,:,1:-1,1:-1]

                        for m in roadType:
                            roadMask = roadMask * (inputs['seman_gt'] != m)
                        roadMask = 1 - roadMask
                        roadMask = roadMask[:,:,1:-1,1:-1]

                        for m in permuType:
                            permuMask = permuMask * (inputs['seman_gt'] != m)
                        permuMask = 1 - permuMask
                        permuMask = permuMask[:,:,1:-1,1:-1]

                    BdErrFig, viewRdErrFig = objReg.visualize_regularizeBuildingRoad(surnormMap, wallMask, roadMask, dispMap, viewInd=index)


                    padSize = int((chanWinSize-1) / 2)
                    permuMask = permuMask[:, :, padSize : -padSize, padSize : -padSize]
                    surVarFig = objReg.visualize_regularizePoleSign(surnormMap, permuMask, dispMap, viewInd=index)

                if viewBorderRegress:
                    foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18] # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
                    backgroundType = [0, 1, 2, 3, 4, 8, 9, 10] # road, sidewalk, building, wall, fence, vegetation, terrain, sky
                    suppressType = [255] # Suppress no label lines
                    # foreGroundMask = torch.sum(inputs['seman_gt'][:, foregroundType, :, :], dim=1, keepdim=True)
                    # backGroundMask = torch.sum(inputs['seman_gt'][:, backgroundType, :, :], dim=1, keepdim=True)
                    foreGroundMask = torch.ones(dispMap.shape).cuda().byte()
                    backGroundMask = torch.ones(dispMap.shape).cuda().byte()
                    suppresMask = torch.ones(dispMap.shape).cuda().byte()

                    with torch.no_grad():
                        for m in foregroundType:
                            foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                        foreGroundMask = 1 - foreGroundMask
                        for m in backgroundType:
                            backGroundMask = backGroundMask * (inputs['seman_gt'] != m)
                        backGroundMask = 1 - backGroundMask
                        for m in suppressType:
                            suppresMask = suppresMask * (inputs['seman_gt'] != m)
                        suppresMask = 1 - suppresMask
                        suppresMask = suppresMask.float()
                        combinedMask = torch.cat([foreGroundMask, backGroundMask], dim=1).float()

                    # borderRegFig = borderRegress.visualize_computeBorder(dispMap, combinedMask, suppresMask = suppresMask, viewIndex=index)
                    borderRegFig = None

                else:
                    borderRegFig = None

                # if viewBorderSimilarity:
                #     foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17,
                #                       18]  # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
                #     backgroundType = [0, 1, 2, 3, 4, 8, 9,
                #                       10]  # road, sidewalk, building, wall, fence, vegetation, terrain, sky
                #     suppressType = [255]  # Suppress no label lines
                #     foreGroundMask = torch.ones(dispMap.shape).cuda().byte()
                #     backGroundMask = torch.ones(dispMap.shape).cuda().byte()
                #     suppresMask = torch.ones(dispMap.shape).cuda().byte()
                #
                #     with torch.no_grad():
                #         for m in foregroundType:
                #             foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                #         foreGroundMask = 1 - foreGroundMask
                #         for m in backgroundType:
                #             backGroundMask = backGroundMask * (inputs['seman_gt'] != m)
                #         backGroundMask = 1 - backGroundMask
                #         for m in suppressType:
                #             suppresMask = suppresMask * (inputs['seman_gt'] != m)
                #         suppresMask = 1 - suppresMask
                #         suppresMask = suppresMask.float()
                #         combinedMask = torch.cat([foreGroundMask, backGroundMask], dim=1).float()
                #
                #     borderSimFig = borderSim.visualize_borderSimilarity(dispMap, foreGroundMask.float(), suppresMask = suppresMask, viewIndex=index)

                if viewRandomSample:
                    foregroundType = [5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18] # pole, traffic light, traffic sign, person, rider, car, truck, bus, train, motorcycle, bicycle
                    backgroundType = [0, 1, 2, 3, 4, 8, 9, 10] # road, sidewalk, building, wall, fence, vegetation, terrain, sky
                    suppressType = [255] # Suppress no label lines
                    foreGroundMask = torch.ones(dispMap.shape).cuda().byte()
                    backGroundMask = torch.ones(dispMap.shape).cuda().byte()
                    suppresMask = torch.ones(dispMap.shape).cuda().byte()

                    with torch.no_grad():
                        for m in foregroundType:
                            foreGroundMask = foreGroundMask * (inputs['seman_gt'] != m)
                        foreGroundMask = 1 - foreGroundMask
                        for m in suppressType:
                            suppresMask = suppresMask * (inputs['seman_gt'] != m)
                        suppresMask = 1 - suppresMask
                        suppresMask = suppresMask.float()
                        foreGroundMask = foreGroundMask.float()

                    rdSampleOnBorder.visualize_randomSample(dispMap, foreGroundMask, suppresMask, viewIndex=index)
                    # rdSampleOnBorder.randomSampleReg(dispMap, foreGroundMask)


                if viewEdgeMerge:
                    grad_disp = comp1dgrad(outputs[('mul_disp', 0)])
                    fig_grad = tensor2disp(grad_disp, ind = index, vmax=1)
                    fig_grad = fig_grad.resize([512, 256])

                if viewSelfOcclu:
                    fl = inputs[("K", 0)][:, 0, 0]
                    bs = torch.abs(inputs["stereo_T"][:, 0, 3])
                    clufig, suppressedDisp = selfclu.visualize(dispMap, viewind=index)


                if fig_grad is not None:
                    grad_seman = (np.array(fig_grad)[:, :, 0:3].astype(np.float) * 0.7 + np.array(fig_seman).astype(np.float) * 0.3).astype(np.uint8)
                    # combined = [np.array(fig_disp)[:, :, 0:3], np.array(fig_grad)[:, :, 0:3], np.array(fig_seman), np.array(fig_rgb)]
                    combined = [grad_seman, np.array(fig_disp)[:, :, 0:3], np.array(fig_rgb)]
                    combined = np.concatenate(combined, axis=1)
                else:
                    if viewSurfaceNormal and viewSelfOcclu:
                        surnorm = surnorm.resize([512, 256])
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
                        comb3 = np.concatenate([np.array(surnorm_mixed)[:, :, 0:3], np.array(surnorm)[:, :, 0:3]], axis=1)
                        comb4 = np.concatenate([np.array(fig_seman)[:, :, 0:3], np.array(rgb_seman)[:, :, 0:3]],
                                               axis=1)
                        comb6 = np.concatenate([np.array(clufig)[:, :, 0:3], np.array(fig_dispup)[:, :, 0:3]], axis=1)

                        fig3dsize = np.ceil(np.array([comb4.shape[1] , comb4.shape[1] / fig_3d.size[0] * fig_3d.size[1]])).astype(np.int)
                        comb5 = np.array(fig_3d.resize(fig3dsize))
                        # combined = np.concatenate([comb1, comb6, comb2, comb3, comb4, comb5], axis=0)
                        combined = np.concatenate([comb1, comb2, comb4, comb3], axis=0)
                    else:
                        disp_seman = (np.array(fig_disp)[:, :, 0:3].astype(np.float) * 0.8 + np.array(fig_seman).astype(np.float) * 0.2).astype(np.uint8)
                        rgb_seman = (np.array(fig_seman).astype(np.float) * 0.5 + np.array(fig_rgb).astype(np.float) * 0.5).astype(np.uint8)
                        # combined = [np.array(disp_seman)[:,:,0:3], np.array(fig_disp)[:, :, 0:3], np.array(fig_seman), np.array(fig_rgb)]
                        combined = [np.array(disp_seman)[:, :, 0:3], np.array(fig_disp)[:, :, 0:3], np.array(fig_seman),
                                    np.array(rgb_seman)]
                        combined = np.concatenate(combined, axis=1)

                fig = pil.fromarray(combined)
                # fig.show()
                fig.save(os.path.join(dirpath, str(idx) + '.png'))
                if borderRegFig is not None:
                    borderRegFig.save(os.path.join(dirpath, str(idx) + '_borderRegress.png'))
                # fig_3d.save(os.path.join(dirpath, str(idx) + '_fig3d.png'))
                # for k in range(10):
                #     fig_disp = tensor2disp(outputs[('disp', 0)], ind=k)
                #     fig_rgb = tensor2rgb(inputs[('color', 0, 0)], ind=k)
                #     combined = [np.array(fig_disp)[:, :, 0:3], np.array(fig_rgb)]
                #     combined = np.concatenate(combined, axis=1)
                #     fig = pil.fromarray(combined)
                #     fig.save(
                #         os.path.join('/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/MoredispOrg' + str(k) + '.png'))



                # fig_rgb.save(os.path.join(svRoot, app, 'rgb' + str(idx) + '.png'))
                # fig_seman.save(os.path.join(svRoot, app, 'semantic'+ str(idx) + '.png'))
                # fig_disp.save(os.path.join(svRoot, app, 'disp'+ str(idx) + '.png'))
                # a = inputs['seman_gt_eval']
                # scaled_disp, _ = disp_to_depth(outputs[('disp', 0)], 0.1, 100)
                print("%dth saved" % idx)
    # If compute the histogram
    if isHist:
        svPath = '/media/shengjie/other/sceneUnderstanding/monodepth2/internalRe/mul_channel_depth'
        carId = 13
        prob = copy.deepcopy(rec)
        ind = np.arange(prob.shape[1] * 2)
        for i in range(prob.shape[0]):
            prob[i,:] = prob[i,:] / np.sum(prob[i,:])
        for i in range(prob.shape[0]):
            trainStr = trainId2label[i][0]
            fig, ax = plt.subplots()
            rects1 = ax.bar(ind[0::2], prob[carId, :], label='obj:car')
            rects2 = ax.bar(ind[1::2], prob[i, :], label='obj:' + trainStr)
            ax.set_ylabel('Meter in percentile')
            ax.set_xlabel('Meters')
            ax.set_title('Scale Changes between scale car and scale %s' % trainStr)
            ax.legend()
            plt.savefig(os.path.join(svPath, str(i)), dpi=200)
            plt.close(fig)



if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())
