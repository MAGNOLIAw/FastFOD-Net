#==============================================================================#
#  Author:       Dominik Müller, Xinyi Wang                                               #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import numpy as np
# Internal libraries/scripts
from processing.subfunctions.abstract_subfunction import Abstract_Subfunction
import nibabel as nib

import os
#-----------------------------------------------------#
#          Subfunction class: Normalization           #
#-----------------------------------------------------#
""" A Normalization Subfunction class which normalizes the intensity pixel values of an image using
    the Z-Score technique (default setting), through scaling to [0,1] or to grayscale [0,255].

Args:
    mode (string):          Mode which normalization approach should be performed.
                            Possible modi: "z-score", "minmax" or "grayscale"

Methods:
    __init__                Object creation function
    preprocessing:          Pixel intensity value normalization the imaging data
    postprocessing:         Do nothing
"""
class Normalization_HCP_SR_FOD(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score"):
        self.mode = mode
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        self.brain_img = None
        self.brain_mask = None
        self.ref_img = None
        self.ref_mask = None
        self.lesion_mask = None

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True, validation=True, opt=None):

        # Access input and mask
        brain_img = sample.img_data.squeeze().copy()
        brain_mask = sample.seg_data.squeeze()

        # IF gt is loaded
        if training or validation:
            ref_img = sample.gt_data.squeeze().copy()
            # ref mask: gt, healthy brain mask
            ref_mask = ref_img.copy()
            ref_mask[ref_mask != 0] = 1

        # Perform z-score normalization
        if self.mode == "z-score":
            # IF use mean and std of valid regions except holes and background (for inpainting)
            mean = np.mean(brain_img[brain_mask == 1])
            std = np.std(brain_img[brain_mask == 1])
            self.mean = mean
            self.std = std
            # Scaling
            brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - mean) / std
            brain_img[brain_mask == 0] = 0
            ref_img[ref_mask == 1] = (ref_img[ref_mask == 1] - mean) / std
            ref_img[ref_mask == 0] = 0
        elif self.mode == 'z-score_v2':
            # IF use mean and standard deviation of brain regions
            if training:
                self.fodlr_mean = np.mean(brain_img[brain_mask == 1], axis=(0))
                self.fodlr_std = np.std(brain_img[brain_mask == 1], axis=(0))
                self.fodgt_mean = np.mean(ref_img[brain_mask == 1], axis=(0))
                self.fodgt_std = np.std(ref_img[brain_mask == 1], axis=(0))
    
                self.fodlr_mean = np.asarray(self.fodlr_mean).reshape(1, 1, 1, -1).astype(np.float32)
                self.fodlr_std = np.asarray(self.fodlr_std).reshape(1, 1, 1, -1).astype(np.float32)
                self.fodgt_mean = np.asarray(self.fodgt_mean).reshape(1, 1, 1, -1).astype(np.float32)
                self.fodgt_std = np.asarray(self.fodgt_std).reshape(1, 1, 1, -1).astype(np.float32)
            else:
                # IF load mean / std of training cases
                checkpoints_dir = opt.checkpoints_dir
                name = opt.name
                gt_mean = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_gt_mean.npy'),
                                    allow_pickle=True)
                gt_std = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_gt_std.npy'), allow_pickle=True)
                lr_mean = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_lr_mean.npy'),
                                    allow_pickle=True)
                lr_std = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_lr_std.npy'), allow_pickle=True)
                self.fodlr_mean = lr_mean
                self.fodlr_std = lr_std
                self.fodgt_mean = gt_mean
                self.fodgt_std = gt_std

                # IF use mean / std of brain regions of low resolution input only
                # self.fodlr_mean = np.mean(brain_img[brain_mask == 1], axis=(0))
                # self.fodlr_std = np.std(brain_img[brain_mask == 1], axis=(0))
                # self.fodgt_mean = self.fodlr_mean
                # self.fodgt_std = self.fodlr_std

            self.mean = self.fodgt_mean
            self.std = self.fodgt_std

            # IF scaling all voxels
            brain_img = (brain_img - self.fodlr_mean) / self.fodlr_std # lr input
            if training or validation:
                ref_img = (ref_img - self.fodgt_mean) / self.fodgt_std # gt
            brain_mask = np.expand_dims(brain_mask, 3).repeat(45, axis=3)
            # IF scaling brain regions only
            # brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - self.fodlr_mean) / self.fodlr_std
            # brain_img[brain_mask == 0] = 0
            # ref_img[ref_mask != 0] = (ref_img[ref_mask == 1] - self.fodgt_mean) / self.fodgt_std
            # ref_img[ref_mask == 0] = 0
        elif self.mode == 'z-score_v3':
            # IF use mean / std of all voxels
            self.fodlr_mean = np.mean(brain_img, axis=(0,1,2)) # a*b*c*45 -> 45
            self.fodlr_std = np.std(brain_img, axis=(0,1,2))
            self.fodgt_mean = np.mean(ref_img, axis=(0,1,2))
            self.fodgt_std = np.std(ref_img, axis=(0,1,2))

            self.fodlr_mean = np.asarray(self.fodlr_mean).reshape(1, 1, 1, -1).astype(np.float32) # 1*1*1*45
            self.fodlr_std = np.asarray(self.fodlr_std).reshape(1, 1, 1, -1).astype(np.float32)
            self.fodgt_mean = np.asarray(self.fodgt_mean).reshape(1, 1, 1, -1).astype(np.float32)
            self.fodgt_std = np.asarray(self.fodgt_std).reshape(1, 1, 1, -1).astype(np.float32)

            self.mean = self.fodgt_mean
            self.std = self.fodgt_std

            # IF scaling all voxels
            brain_img = (brain_img - self.fodlr_mean) / self.fodlr_std # a,b,c,45 - 1,1,1,45
            ref_img = (ref_img - self.fodgt_mean) / self.fodgt_std

            brain_mask = np.expand_dims(brain_mask, 3).repeat(45, axis=3)

        elif self.mode == 'z-score_v4':
            # IF use mean and standard deviation of brain regions
            if training:
                self.fodlr_mean = np.mean(brain_img[brain_mask == 1], axis=(0)) # 45,
                self.fodlr_std = np.std(brain_img[brain_mask == 1], axis=(0))
                # self.fodgt_mean = np.mean(ref_img[brain_mask == 1], axis=(0))
                # self.fodgt_std = np.std(ref_img[brain_mask == 1], axis=(0))
                self.fodgt_mean = self.fodlr_mean
                self.fodgt_std = self.fodlr_std

                self.fodlr_mean = np.asarray(self.fodlr_mean).reshape(1, 1, 1, -1).astype(np.float32)
                self.fodlr_std = np.asarray(self.fodlr_std).reshape(1, 1, 1, -1).astype(np.float32)
                self.fodgt_mean = np.asarray(self.fodgt_mean).reshape(1, 1, 1, -1).astype(np.float32)
                self.fodgt_std = np.asarray(self.fodgt_std).reshape(1, 1, 1, -1).astype(np.float32) # 1, 1, 1, 45
            else:
                # IF load mean / std of training cases
                # gt_mean = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_gt_mean.npy'),
                #                     allow_pickle=True)
                # gt_std = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_gt_std.npy'), allow_pickle=True)
                # lr_mean = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_lr_mean.npy'),
                #                     allow_pickle=True)
                # lr_std = np.load(os.path.join(checkpoints_dir, name, 'z-score_v2' + '_lr_std.npy'), allow_pickle=True)
                # self.fodlr_mean = lr_mean
                # self.fodlr_std = lr_std
                # self.fodgt_mean = gt_mean
                # self.fodgt_std = gt_std

                # IF use mean / std of brain regions of low resolution input only
                self.fodlr_mean = np.mean(brain_img[brain_mask == 1], axis=(0))
                self.fodlr_std = np.std(brain_img[brain_mask == 1], axis=(0))
                # before 01/06/24
                self.fodgt_mean = self.fodlr_mean
                self.fodgt_std = self.fodlr_std
                # after
                # self.fodgt_mean = np.asarray(self.fodgt_mean).reshape(1, 1, 1, -1).astype(np.float32)
                # self.fodgt_std = np.asarray(self.fodgt_std).reshape(1, 1, 1, -1).astype(np.float32) # 1, 1, 1, 45

            self.mean = self.fodgt_mean
            self.std = self.fodgt_std

            # IF scaling all voxels
            brain_img = (brain_img - self.fodlr_mean) / self.fodlr_std  # lr input
            if training or validation:
                ref_img = (ref_img - self.fodgt_mean) / self.fodgt_std  # gt
            brain_mask = np.expand_dims(brain_mask, 3).repeat(45, axis=3)

        elif self.mode == "notnorm":
            brain_img = brain_img
            ref_img = ref_img

        # Perform MinMax normalization between [0,1]
        elif self.mode == "minmax":
            # Identify minimum and maximum
            min_val = np.min(brain_img[brain_mask == 1])
            max_val = np.max(brain_img[brain_mask == 1])
            val_range = max_val - min_val
            self.min = min_val
            self.max = max_val
            # Scaling
            brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - min_val) / val_range
            brain_img[brain_mask == 0] = 0
            ref_img[ref_mask != 0] = (ref_img[ref_mask == 1] - min_val) / val_range
            ref_img[ref_mask == 0] = 0

            # image_normalized = brain_img
            image_normalized = ref_img

        elif self.mode == "grayscale":
            # Identify minimum and maximum
            max_value = np.max(brain_img[brain_mask == 1])
            min_value = np.min(brain_img[brain_mask == 1])
            # Scaling
            ref_img[ref_mask != 0] = (ref_img[ref_mask == 1] - min_value) / (max_value - min_value)
            ref_img[ref_mask == 0] = 0
            image_scaled = ref_img
            image_normalized = np.around(image_scaled * 255, decimals=0)

        else : raise NameError("Subfunction - Normalization: Unknown modus")

        # Update the sample with the normalized image
        sample.img_data = brain_img
        sample.seg_data = brain_mask
        self.brain_img = brain_img
        self.brain_mask = brain_mask
        if training or validation:
            sample.gt_data = ref_img
            self.ref_img = ref_img
            self.ref_mask = ref_mask
        self.sample = sample
    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        return prediction
