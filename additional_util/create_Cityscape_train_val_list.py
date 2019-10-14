import glob
import numpy as np
import os
import random
def generateCityScapeSplit(datasetLoc, splitFileLoc):
    kittiLength = 45200 # maintain the other file is around 10000
    to_expand = 10000
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    boostTime = np.ceil(to_expand / len(fineList))
    blendedList = fineList * np.int(boostTime)
    random.shuffle(blendedList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()
    val_fineList = list()
    kittiValLength = 4424
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            val_fineList.append(imagePath)

    boostTime = kittiValLength / len(val_fineList)
    blendedList = val_fineList * np.int(boostTime)
    random.shuffle(blendedList)

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
    fileVal.close()
def generateCityScape_scaleVerification_split(datasetLoc, splitFileLoc):
    # This is only served for generating validation file for scale adjustment
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            # if np.random.random(1) > 0.5:
            #     writeComp3 = 'l'
            # else:
            #     writeComp3 = 'r'
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
            index = index + 1
    fileTrain.close()

def generateCityScapeSplitExtra(datasetLoc, splitFileLoc, splitFileNum):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    # Balance fine and coarse label

    fineList = list()
    coarseList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train_extra", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            coarseList.append(imagePath)
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        # index = 0 # continues index
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    boostTime = np.ceil(len(coarseList) / 2 / len(fineList))
    blendedList = fineList * np.int(boostTime) + coarseList
    random.shuffle(blendedList)

    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
            index = index + 1
    fileVal.close()
def generateCityScapeSplit_riginalSize(datasetLoc, splitFileLoc):
    kittiLength = 45200 # maintain the other file is around 10000
    to_expand = 10000
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    boostTime = 1
    blendedList = fineList * np.int(boostTime)
    random.shuffle(blendedList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
            index = index + 1
    fileVal.close()

def generateCityScapeSplit(datasetLoc, splitFileLoc):
    kittiLength = 45200
    to_expand = 5000
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    boostTime = np.ceil(to_expand / len(fineList))
    blendedList = fineList * np.int(boostTime)
    random.shuffle(blendedList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()

    val_fineList = list()
    kittiValLength = 4424
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            val_fineList.append(imagePath)

    boostTime = kittiValLength / len(val_fineList)
    blendedList = val_fineList * np.int(boostTime)
    random.shuffle(blendedList)

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
    fileVal.close()

def cityscapeCoarseOnly(datasetLoc, splitFileLoc):
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train_extra", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    blendedList = fineList
    random.shuffle(blendedList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()

    val_fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            val_fineList.append(imagePath)

    blendedList = val_fineList
    random.shuffle(blendedList)

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
    fileVal.close()


def cityscapeFine2CoarseOnly(datasetLoc, splitFileLoc):
    boostTime = int(np.ceil(20000/4000))
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    blendedList = fineList * boostTime
    random.shuffle(blendedList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()

    val_fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            val_fineList.append(imagePath)

    blendedList = val_fineList
    random.shuffle(blendedList)

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
    fileVal.close()
def createToyExample(datasetLoc, splitFileLoc):
    boostTime = 1
    fineList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train_extra", "*")):
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    random.shuffle(fineList)
    fineList = fineList[0:1000]
    fineList = fineList * boostTime
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for index, imagePath in enumerate(fineList):
        split_comp = imagePath.split("/")
        writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
        writeComp2 = split_comp[-1]
        writeComp3 = 'l'
        writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
            writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
        fileTrain.writelines(writel)
    fileTrain.close()

    # val_fineList = fineList
    # for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
    #     for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
    #         val_fineList.append(imagePath)

    # blendedList = val_fineList
    # random.shuffle(blendedList)

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for index, imagePath in enumerate(fineList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
    fileVal.close()


def generateCityScapeSplitExtraLR(datasetLoc, splitFileLoc):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    # Balance fine and coarse label

    fineList = list()
    coarseList = list()
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train_extra", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            coarseList.append(imagePath)
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "train", "*")):
        # index = 0 # continues index
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            fineList.append(imagePath)
    # boostTime = np.ceil(len(coarseList) / len(fineList))
    boostTime = 1
    blendedList = fineList * np.int(boostTime) + coarseList
    random.shuffle(blendedList)

    for index, imagePath in enumerate(blendedList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            if np.random.rand(1) > 0.5:
                writeComp3 = 'l'
            else:
                writeComp3 = 'r'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileTrain.writelines(writel)
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for subFolder in glob.glob(os.path.join(datasetLoc, "leftImg8bit", "val", "*")):
        index = 0
        for imagePath in glob.glob(os.path.join(subFolder, "*.png")):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0][0:len(writeComp2.split('.')[0]) - len(
                writeComp2.split('.')[0].split('_')[-1])] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
            index = index + 1
    fileVal.close()

if __name__ == "__main__":
    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape"
    # generateCityScapeSplit(datasetLoc, splitFileLoc)

    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape_verification_scale"
    # generateCityScape_scaleVerification_split(datasetLoc, splitFileLoc)

    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscapeExtra"
    # generateCityScapeSplitExtra(datasetLoc, splitFileLoc)

    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape_original"
    # generateCityScapeSplit_riginalSize(datasetLoc, splitFileLoc)

    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscapeCoarseOnly"
    # cityscapeCoarseOnly(datasetLoc, splitFileLoc)


    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscapeFine2CoarseOnly"
    # cityscapeFine2CoarseOnly(datasetLoc, splitFileLoc)

    # datasetLoc = "/media/shengjie/other/cityscapesData"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/cityscape_toy"
    # createToyExample(datasetLoc, splitFileLoc)



    datasetLoc = "/media/shengjie/other/cityscapesData"
    splitFileLoc = "/media/shengjie/other/sceneUnderstanding/SDNET/splits/cityscape"
    splitFileNum = 30000
    splitFileLoc = splitFileLoc + str(splitFileNum)
    generateCityScapeSplitExtra(datasetLoc, splitFileLoc, splitFileNum)