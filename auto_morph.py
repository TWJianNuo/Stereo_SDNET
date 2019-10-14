from numba import jit
from scipy.spatial import Delaunay
import numpy as np
import PIL.Image as pil
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from utils import *
import cv2
import torch.nn as nn

@jit(nopython=True)
def searchNearestPts(src_bin, dst_bin, height, width, srcxy, dstxy, validPts, colSearchOrd, rowSearchOrd, allowedRadius, ckTable):
    count = 0
    for i in range(height):
        for j in range(width):
            if src_bin[i,j] == 1:
                srcxy[count, 0] = j
                srcxy[count, 1] = i
                for m in colSearchOrd:
                    for n in rowSearchOrd:
                        colSearchInd = i + m
                        rowSearchInd = j + n

                        if ckTable[i, j] == 0:
                            if colSearchInd >= 0 and colSearchInd < height and rowSearchInd >=0 and rowSearchInd < width:
                                if dst_bin[colSearchInd, rowSearchInd] == 1:
                                    dstxy[count, 0] = rowSearchInd
                                    dstxy[count, 1] = colSearchInd
                                    validPts[count] = 1

                                    for p in range(-allowedRadius, allowedRadius + 1):
                                        for q in range(-allowedRadius, allowedRadius + 1):
                                            colFillInd = i + p
                                            rowFillInd = j + q
                                            if colFillInd >= 0 and colFillInd < height and rowFillInd >= 0 and rowFillInd < width:
                                                ckTable[colFillInd, rowFillInd] = 1

                                    break
                count = count + 1
@jit(nopython=True)
def organizeMeshGrid(ptsSrc, ptsDst, gridLookUpTable, gridValidRec, gridDense, itNum):
    for i in range(itNum):
        srcx = np.int(np.floor(ptsSrc[i, 0] / gridDense))
        srcy = np.int(np.floor(ptsSrc[i, 1] / gridDense))
        dstx = np.int(np.floor(ptsDst[i, 0] / gridDense))
        dsty = np.int(np.floor(ptsDst[i, 1] / gridDense))

        if srcx <= dstx:
            smallx = srcx
            bigx = dstx
        else:
            smallx = dstx
            bigx = srcx

        if srcy <= dsty:
            smally = srcy
            bigy = dsty
        else:
            smally = dsty
            bigy = srcy

        for m in range(smally, bigy + 1):
            for n in range(smallx, bigx + 1):
                if n < bigx:
                    invalLinInd1 = gridLookUpTable[m, n + 1]
                    invalLinInd2 = gridLookUpTable[m + 1, n + 1]
                    gridValidRec[invalLinInd1] = 0
                    gridValidRec[invalLinInd2] = 0
                if m < bigy:
                    invalLinInd1 = gridLookUpTable[m + 1, n]
                    invalLinInd2 = gridLookUpTable[m + 1, n + 1]
                    gridValidRec[invalLinInd1] = 0
                    gridValidRec[invalLinInd2] = 0

        if np.mod(ptsSrc[i, 0], gridDense) == 0:
            invalLinInd1 = gridLookUpTable[srcy, srcx]
            invalLinInd2 = gridLookUpTable[srcy + 1, srcx]
            gridValidRec[invalLinInd1] = 0
            gridValidRec[invalLinInd2] = 0

        if np.mod(ptsSrc[i, 1], gridDense) == 0:
            invalLinInd1 = gridLookUpTable[srcy, srcx]
            invalLinInd2 = gridLookUpTable[srcy, srcx + 1]
            gridValidRec[invalLinInd1] = 0
            gridValidRec[invalLinInd2] = 0


class grad_computation_tools(nn.Module):
    def __init__(self, batch_size, height, width):
        super(grad_computation_tools, self).__init__()
        weightsx = torch.Tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)

        weightsy = torch.Tensor([
            [1., 2., 1.],
            [0., 0., 0.],
            [-1., -2., -1.]]).unsqueeze(0).unsqueeze(0)
        self.convDispx = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispy = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.convDispx.weight = nn.Parameter(weightsx, requires_grad=False)
        self.convDispy.weight = nn.Parameter(weightsy, requires_grad=False)

        self.disparityTh = 0.011
        self.semanticsTh = 0.6

        self.zeroRange = 2
        self.zero_mask = torch.ones([batch_size, 1, height, width]).cuda()
        self.zero_mask[:, :, :self.zeroRange, :] = 0
        self.zero_mask[:, :, -self.zeroRange:, :] = 0
        self.zero_mask[:, :, :, :self.zeroRange] = 0
        self.zero_mask[:, :, :, -self.zeroRange:] = 0

        self.mask = torch.ones([batch_size, 1, height, width], device=torch.device("cuda"))
        self.mask[:,:,0:128,:] = 0
    def get_dradient(self, disparityMap, foregroundMapGt):
        disparity_grad = torch.abs(self.convDispx(disparityMap)) + torch.abs(self.convDispy(disparityMap))
        semantics_grad = torch.abs(self.convDispx(foregroundMapGt)) + torch.abs(self.convDispy(foregroundMapGt))
        disparity_grad = disparity_grad * self.zero_mask
        semantics_grad = semantics_grad * self.zero_mask
class AutoMorph():
    def __init__(self, height, width):
        self.searchWidth = 7
        self.searchHeight = 3
        self.allowedRadius = 3
        self.gridDense = 10

        colsearchSpan = np.arange(-self.searchHeight, self.searchHeight + 1)
        colsearchInd = np.argsort(np.abs(colsearchSpan))
        self.colSearchOrd = colsearchSpan[colsearchInd]

        rowsearchSpan = np.arange(-self.searchWidth, self.searchWidth + 1)
        rowsearchInd = np.argsort(np.abs(rowsearchSpan))
        self.rowSearchOrd = rowsearchSpan[rowsearchInd]



        gridxs = np.arange(0, width, self.gridDense)
        if np.mod(width - 1, self.gridDense) > 0:
            gridxs = np.append(gridxs, width - 1)
        gridys = np.arange(0, height, self.gridDense)
        if np.mod(height - 1, self.gridDense) > 0:
            gridys = np.append(gridys, height - 1)
        xx, yy = np.meshgrid(gridxs, gridys)


        self.gridLookUpTable = np.arange(0, xx.shape[0] * xx.shape[1]).reshape(xx.shape[0], xx.shape[1])
        # check
        # flatxck = xx.flatten()
        # flatyck = yy.flatten()
        # rndx = np.random.randint(0, xx.shape[1])
        # rndy = np.random.randint(0, xx.shape[0])
        # serialInd = self.gridLookUpTable[rndy, rndx]
        # serialx = flatxck[serialInd]
        # serialy = flatyck[serialInd]
        # spacialx = xx[rndy, rndx]
        # spacialy = yy[rndy, rndx]
        # diff = (serialx - spacialx)**2 + (serialy - spacialy)**2

        self.xx = xx.flatten()
        self.yy = yy.flatten()
        self.height = height
        self.width = width

    def applyAffineTransform(self, src, srcTri, dstTri, size):
        # Given a pair of triangles, find the affine transform.
        warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

        # Apply the Affine Transform just found to the src image
        dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)

        return dst
    def morphSingleTriangle(self, imgsrc, imgdst, tsrc, tdst, changeingRec):
        rsrc = cv2.boundingRect(np.float32([tsrc]))
        rdst = cv2.boundingRect(np.float32([tdst]))

        # Offset points by left top corner of the respective rectangles
        dstRect = []
        srcRect = []

        for i in range(0, 3):
            srcRect.append(((tsrc[i][0] - rsrc[0]), (tsrc[i][1] - rsrc[1])))
            dstRect.append(((tdst[i][0] - rdst[0]), (tdst[i][1] - rdst[1])))

        # Get mask by filling triangle
        mask = np.zeros((rdst[3], rdst[2]), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dstRect), (1.0), 16, 0)

        # Apply warpImage to small rectangular patches
        imgRect = imgsrc[rsrc[1]:rsrc[1] + rsrc[3], rsrc[0]:rsrc[0] + rsrc[2]]

        size = (rdst[2], rdst[3])
        warpImage = self.applyAffineTransform(imgRect, srcRect, dstRect, size)


        # Copy triangular region of the rectangular patch to the output image
        imgdst[rdst[1]:rdst[1] + rdst[3], rdst[0]:rdst[0] + rdst[2]] = imgdst[rdst[1]:rdst[1]+rdst[3], rdst[0]:rdst[0]+rdst[2]] * ( 1 - mask ) + warpImage * mask
        changeingRec[rdst[1]:rdst[1] + rdst[3], rdst[0]:rdst[0] + rdst[2]] += mask
    def morphTiagngles(self, triangleRef, ptssrc, ptsdst, dispMapsrc, dispMapdst, changeingRec):
        for i in range(triangleRef.shape[0]):
            srcTriangle = ptssrc[triangleRef[i,:]]
            dstTriangle = ptsdst[triangleRef[i,:]]
            self.morphSingleTriangle(imgsrc=dispMapsrc, imgdst=dispMapdst, tsrc=srcTriangle, tdst=dstTriangle, changeingRec = changeingRec)
    def automorph(self, src_bin, dst_bin, dispMap):
        height = self.height
        width = self.width
        totPts = np.sum(src_bin)
        srcxy = np.zeros((totPts, 2), dtype=np.int)
        dstxy = np.zeros((totPts, 2), dtype=np.int)
        validPts = np.zeros(totPts, dtype=np.bool)
        ckTable = np.zeros([height, width], dtype=np.bool)

        dispMap_morphed = np.copy(dispMap)
        changeingRec = np.zeros_like(dispMap_morphed)

        searchNearestPts(src_bin = src_bin, dst_bin = dst_bin,
                         height=height, width=width,
                         srcxy=srcxy, dstxy=dstxy,
                         validPts=validPts,
                         colSearchOrd=self.colSearchOrd, rowSearchOrd=self.rowSearchOrd, allowedRadius=self.allowedRadius, ckTable=ckTable)

        # synthesizedRGB = np.stack([src_bin, dst_bin, np.zeros_like(dst_bin)], axis=2)
        # synthesizedRGB = (synthesizedRGB * 255).astype(np.uint8)
        # synthesizedRGB_fig = pil.fromarray(synthesizedRGB)
        # plt.figure()
        # plt.imshow(synthesizedRGB)
        # srcpts = srcxy[validPts, :]
        # dstPts = dstxy[validPts, :]
        # for i in range(0, srcpts.shape[0]):
        #     plt.plot([srcpts[i, 0], dstPts[i, 0]], [srcpts[i, 1], dstPts[i, 1]])
        # plt.show()
        # plt.close()
        if np.sum(validPts) == 0:
            return dispMap_morphed, changeingRec
        srcxy = srcxy[validPts, :]
        dstxy = dstxy[validPts, :]
        gridValidRec = np.ones(self.xx.shape[0], dtype=np.bool)
        organizeMeshGrid(ptsSrc=srcxy, ptsDst=dstxy, gridLookUpTable=self.gridLookUpTable, gridValidRec=gridValidRec, gridDense=self.gridDense, itNum=srcxy.shape[0])

        # drawGridx = self.xx[gridValidRec]
        # drawGridy = self.yy[gridValidRec]
        # plt.figure()
        # for i in range(0, srcpts.shape[0]):
        #     plt.plot([srcpts[i, 0], dstPts[i, 0]], [srcpts[i, 1], dstPts[i, 1]])
        # plt.scatter(drawGridx, drawGridy, s = 1, c = 'r')
        # plt.scatter(srcxy[:, 0], srcxy[:, 1], s=1, c='g')
        # plt.show()
        # plt.close()

        validGridPts = np.stack([self.xx[gridValidRec], self.yy[gridValidRec]], axis=1)
        delaunay_src = np.concatenate([validGridPts, srcxy], axis=0)
        delaunay_dst = np.concatenate([validGridPts, dstxy], axis=0)
        tri = Delaunay(delaunay_src)

        # plt.figure()
        # img1 = pil.open("/media/shengjie/other/sceneUnderstanding/SDNET/1.png")
        # img2 = pil.open("/media/shengjie/other/sceneUnderstanding/SDNET/2.png")
        # img_overlayed = pil.fromarray((np.array(img1) * 0.3 + np.array(img2) * 0.7).astype(np.uint8))

        # plt.imshow(img_overlayed)
        # plt.triplot(delaunay_src[:,0], delaunay_src[:,1], tri.simplices.copy(), linewidth = 0.5)
        # plt.scatter(delaunay_src[:, 0], delaunay_src[:, 1], s=0.5, c='g')
        # plt.show()
        # plt.close()

        # plt.figure()
        # plt.imshow(img_overlayed)
        # plt.triplot(delaunay_dst[:,0], delaunay_dst[:,1], tri.simplices.copy(), linewidth = 0.5)
        # plt.scatter(delaunay_dst[:, 0], delaunay_dst[:, 1], s=0.5, c='g')
        # plt.show()
        # plt.close()


        gridPtsNum = validGridPts.shape[0]
        changedTriangles_ref = np.sum(tri.simplices >= gridPtsNum, axis=1) > 0
        changedTriangles = tri.simplices[changedTriangles_ref, :]
        self.morphTiagngles(triangleRef=changedTriangles, ptssrc=delaunay_src, ptsdst=delaunay_dst, dispMapsrc=dispMap, dispMapdst=dispMap_morphed, changeingRec = changeingRec)
        # fig_disp_org = visualizeNpDisp(dispMap, vmax=0.1)
        # fig_disp_morphed = visualizeNpDisp(dispMap_morphed, vmax=0.1)
        # img_overlayed = pil.fromarray((np.array(img1) * 0.3 + np.array(fig_disp_morphed) * 0.7).astype(np.uint8))
        return dispMap_morphed, changeingRec