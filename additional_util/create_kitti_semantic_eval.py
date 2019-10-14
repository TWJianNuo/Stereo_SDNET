import glob
import numpy as np
import os
import random
import argparse
def generateKittySemanticSplit(datasetLoc, splitFileLoc):
    val_fineList = list()
    for imagePath in glob.glob(os.path.join(datasetLoc, "training", "image_2", "*")):
        val_fineList.append(imagePath)
    random.shuffle(val_fineList)
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for index, imagePath in enumerate(val_fineList):
            split_comp = imagePath.split("/")
            writeComp1 = os.path.join(split_comp[-3], split_comp[-2])
            writeComp2 = split_comp[-1]
            writeComp3 = 'l'
            writel = writeComp1 + '/' + writeComp2.split('.')[0] + " " + format(index, '010') + " " + writeComp3 + "\n"
            fileVal.writelines(writel)
    fileVal.close()
def generateKittiSemanDepthSplit(mappingFileLoc, splitFileLoc):
    with open('/media/shengjie/other/sceneUnderstanding/monodepth2/splits/train_mapping.txt') as f:
        mapping = f.readlines()

    mapping = [x.strip() for x in mapping]
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for line in mapping:
        if len(line) > 1:
            lineComp = line.split(' ')
            writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
            fileTrain.writelines(writel)
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for line in mapping:
        if len(line) > 1:
            lineComp = line.split(' ')
            writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
            fileVal.writelines(writel)
    fileVal.close()

def generateKittiToyExaple(mappingFileLoc, splitFileLoc):
    repeatTime = 5000
    with open('/media/shengjie/other/sceneUnderstanding/monodepth2/splits/train_mapping.txt') as f:
        mapping = f.readlines()

    mapping = [x.strip() for x in mapping]
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    for i in range(repeatTime):
        line = mapping[2]
        lineComp = line.split(' ')
        if random.random() > 0.5:
            writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
        else:
            writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'r\n'
        fileTrain.writelines(writel)
    fileTrain.close()

    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")
    for i in range(repeatTime):
        line = mapping[2]
        lineComp = line.split(' ')
        writel = lineComp[0] + '/' + lineComp[1] + ' ' + str(int(lineComp[2])) + ' ' + 'l\n'
        fileVal.writelines(writel)
    fileVal.close()
def generateKittiWithSemanticPredictions(splitFileLoc, kitti_dataset_root):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")

    img_dir_names = glob.glob(kitti_dataset_root + '/*/')
    for img_dir_name in img_dir_names:
        img_subdir_names = glob.glob(os.path.join(kitti_dataset_root, img_dir_name) + '/*/')
        for img_subdir_name in img_subdir_names:
            inspected_rng_02 = os.path.join(kitti_dataset_root, img_dir_name, img_subdir_name, 'image_02', 'data')
            inspected_rng_03 = os.path.join(kitti_dataset_root, img_dir_name, img_subdir_name, 'image_03', 'data')
            for rgb_path in glob.glob(os.path.join(inspected_rng_02, '*.png')):
                semantic_rgb_path = rgb_path.replace('image_02/data', 'semantic_prediction/image_02')
                if os.path.isfile(semantic_rgb_path):
                    path_components = semantic_rgb_path.split('/')
                    to_write = os.path.join(path_components[-5], path_components[-4]) + ' ' + str(int(path_components[-1].split('.')[0])) + ' ' + 'l' + '\n'
                    fileTrain.writelines(to_write)
                    fileVal.writelines(to_write)
            for rgb_path in glob.glob(os.path.join(inspected_rng_03, '*.png')):
                semantic_rgb_path = rgb_path.replace('image_03/data', 'semantic_prediction/image_03')
                if os.path.isfile(semantic_rgb_path):
                    path_components = semantic_rgb_path.split('/')
                    to_write = os.path.join(path_components[-5], path_components[-4]) + ' ' + str(int(path_components[-1].split('.')[0])) + ' ' + 'l' + '\n'
                    fileTrain.writelines(to_write)
                    fileVal.writelines(to_write)
    fileTrain.close()
    fileVal.close()

def create_kitti_semantic_map(splitFileLoc, enalrgeTimes):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")

    with open('/media/shengjie/other/sceneUnderstanding/SDNET/splits/kitti_seman_mapped2depth/train_files.txt') as f:
        content = f.readlines()
    for rgb_path in content:
        fileVal.writelines(rgb_path)
    boostTimes = int(enalrgeTimes / len(content))
    content = content * boostTimes
    random.shuffle(content)

    for rgb_path in content:
        fileTrain.writelines(rgb_path)
    fileTrain.close()
    fileVal.close()

def create_eigen_left_right(splitFileLoc, enalrgeTimes):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")

    lrmap = {'l' : 'r', 'r' : 'l'}
    with open('/media/shengjie/other/sceneUnderstanding/SDNET/splits/eigen_zhou/train_files.txt') as f:
        content = f.readlines()
    for rgb_path in content:
        rgb_path_opp = list(rgb_path)
        rgb_path_opp[-2] = lrmap[rgb_path[-2]]
        rgb_path_opp = ''.join(rgb_path_opp)
        fileTrain.writelines(rgb_path)
        fileTrain.writelines(rgb_path_opp)
    with open('/media/shengjie/other/sceneUnderstanding/SDNET/splits/eigen_zhou/val_files.txt') as f:
        content = f.readlines()
    for rgb_path in content:
        fileVal.writelines(rgb_path)
    fileTrain.close()
    fileVal.close()


parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--kitti_dataset_root', type=str)
parser.add_argument('--splitFileLoc', type=str)
args = parser.parse_args()

def create_kitti3000_left_right(splitFileLoc, enalrgeTimes):
    fileTrain = open(os.path.join(splitFileLoc, "train_files.txt"), "w+")
    fileVal = open(os.path.join(splitFileLoc, "val_files.txt"), "w+")

    lrmap = {'l' : 'r', 'r' : 'l'}
    with open('/media/shengjie/other/sceneUnderstanding/SDNET/splits/eigen_zhou/train_files.txt') as f:
        content = f.readlines()
    random.shuffle(content)
    content = content[0:3000]

    for rgb_path in content:
        rgb_path_opp = list(rgb_path)
        rgb_path_opp[-2] = lrmap[rgb_path[-2]]
        rgb_path_opp = ''.join(rgb_path_opp)
        fileTrain.writelines(rgb_path)
        fileTrain.writelines(rgb_path_opp)
    with open('/media/shengjie/other/sceneUnderstanding/SDNET/splits/eigen_zhou/val_files.txt') as f:
        content = f.readlines()
    for rgb_path in content:
        fileVal.writelines(rgb_path)
    fileTrain.close()
    fileVal.close()


parser = argparse.ArgumentParser(description='evaluation')
parser.add_argument('--kitti_dataset_root', type=str)
parser.add_argument('--splitFileLoc', type=str)
args = parser.parse_args()


if __name__ == "__main__":
    # datasetLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/kitti_semantic_eval"
    # generateKittiSemanDepthSplit(datasetLoc, splitFileLoc)

    # mappingFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/kitti_seman_mapped2depth"
    # generateKittiSemanDepthSplit(mappingFileLoc, splitFileLoc)

    # mappingFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/kitti_data/kitti_semantics"
    # splitFileLoc = "/media/shengjie/other/sceneUnderstanding/monodepth2/splits/kitti_seman_mapped_toy"
    # generateKittiToyExaple(mappingFileLoc, splitFileLoc)
    # generateKittiWithSemanticPredictions(args.splitFileLoc, args.kitti_dataset_root)

    boostTime = 30000
    create_kitti3000_left_right('/media/shengjie/other/sceneUnderstanding/SDNET/splits/kitti3000_lr', boostTime)