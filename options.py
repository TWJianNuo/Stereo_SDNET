# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp"))
        # TRAINING options
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 default="eigen_zhou")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti")
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=288) # previous is 192
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=960) # previous is 640
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
        self.parser.add_argument("--semantic_minscale",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 choices=[0,1,2,3],
                                 default=[3])
        self.parser.add_argument("--backBone",
                                 nargs="+",
                                 type=str,
                                 help="backbone used",
                                 choices=["unet", "ASPP"],
                                 default=["unet"])
        self.parser.add_argument("--is_sep_train_seman",
                                 help="whether to use different training image for semantic segmentation and deoth",
                                 action="store_true"
                                 )
        self.parser.add_argument("--load_meta",
                                 help="load meta data, for cityscape load gt depth, for kitti load semantics data if have",
                                 action="store_true"
                                 )
        self.parser.add_argument("--val_frequency",
                                 type=int,
                                 default=10,
                                 help="set evaluation frequency"
                                 )
        self.parser.add_argument("--stereo_results_path",
                                 type=str
                                 )
        self.parser.add_argument("--read_stereo",
                                 action="store_true"
                                 )
        self.parser.add_argument("--use_two_images",
                                 action="store_true"
                                 )
        self.parser.add_argument("--use_mask_input",
                                 action="store_true"
                                 )
        self.parser.add_argument("--direction_left",
                                 action="store_true"
                                 )
        self.parser.add_argument("--outputtwoimage",
                                 action="store_true"
                                 )
        self.parser.add_argument("--eval_masked_prediction",
                                 action="store_true"
                                 )
        self.parser.add_argument("--outputvisualizaiton",
                                 action="store_true"
                                 )

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--writeImg",
                                 action="store_true")
        self.parser.add_argument("--debug",
                                 action="store_true")


        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--switchMode",
                                 type=str,
                                 help="turn on switch mode or not",
                                 default="off",
                                 choices=["on", "off"])
        self.parser.add_argument("--semanticCoeff",
                                 type=float,
                                 help="Change semantic loss ratio",
                                 default=1.0,)
        self.parser.add_argument("--isMulChannel",
                                 help="if set, Use multiple depth channel",
                                 action="store_true")
        self.parser.add_argument("--banSemantic",
                                 help="if set, Forbit predict semantic segmentation",
                                 action="store_true")
        self.parser.add_argument("--banDepth",
                                 help="if set, Forbit predict depth",
                                 action="store_true")
        self.parser.add_argument("--selfocclu",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--secondOrderSmooth",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--secondOrderSmoothScale",
                                 help="if set, use self occlusion in training",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--predictboth",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--borderMergeLoss",
                                 help="if set, use self occlusion in training",
                                 action="store_true")
        self.parser.add_argument("--borderMergeLossScale",
                                 type=float,
                                 default=1.0
                                 )
        self.parser.add_argument("--use_kitti_gt_semantics",
                                 help="if set, use kitti gt smeantics",
                                 action="store_true")
        self.parser.add_argument("--borderMorphLoss",
                                 help="if set, use border morphing loss",
                                 action="store_true")
        self.parser.add_argument("--borderMorphScale",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--is_predicted_semantics",
                                 action="store_true")
        self.parser.add_argument("--is_toymode",
                                 action="store_true")
        self.parser.add_argument("--borderMorphTime",
                                 type=int,
                                 default=5)
        self.parser.add_argument("--normalTrainingTime",
                                 type=int,
                                 default=5)
        self.parser.add_argument("--load_morphed_depth",
                                 action="store_true")
        self.parser.add_argument("--gaussian_sigma",
                                 type=float,
                                 default=-1)
                                 # default=0.5)
        self.parser.add_argument("--inline_finetune",
                                 action="store_true")
        self.parser.add_argument("--isInsNorm",
                                 action="store_true")
        self.parser.add_argument("--isAff",
                                 action="store_true")
        self.parser.add_argument("--isCudaMorphing",
                                 action="store_true")
        self.parser.add_argument("--l1_weight",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--is_photometric_finetune",
                                 action="store_true")
        self.parser.add_argument("--use_ssim_compare_mask",
                                 action="store_true")
        self.parser.add_argument("--group_two",
                                 action="store_true")
        self.parser.add_argument("--more_sv",
                                 action="store_true")
        self.parser.add_argument("--delay_open",
                                 action="store_true")
        self.parser.add_argument("--seman_reg",
                                 action="store_true")
        self.parser.add_argument("--seman_reg_weight",
                                 type=float,
                                 default=1.0)
        self.parser.add_argument("--lr_consistence",
                                 action="store_true")
        self.parser.add_argument("--no_train_shuffle",
                                 action="store_true")
        self.parser.add_argument("--de_flying_blob",
                                 action="store_true")
        self.parser.add_argument("--de_flying_blob_weight",
                                 type=float,
                                 default=0.1)
        self.parser.add_argument("--concrete_reg",
                                 action="store_true")
        self.parser.add_argument("--concrete_reg_param",
                                 type=float,
                                 default=0.005)
        self.parser.add_argument("--lr_regularization",
                                 action="store_true")
        self.parser.add_argument("--lr_reg_weight",
                                 type=float,
                                 default=1
                                 )
        self.parser.add_argument("--output_dir",
                                 type=str
                                 )
        self.parser.add_argument("--split_name",
                                 type=str
                                 )
        self.parser.add_argument("--appendix_name",
                                 type=str
                                 )
        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--load_weights_folders",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth"])
        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

        # Visualization
        self.parser.add_argument("--view_right",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
