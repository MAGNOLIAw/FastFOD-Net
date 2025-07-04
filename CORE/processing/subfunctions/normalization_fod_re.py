#==============================================================================#  
#  Authors:     Xinyi Wang                                                     #
#  License:     GNU GPL v3.0                                                   #
#==============================================================================#
from utils.logger import setup_logger
logger = setup_logger()

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
class Normalization_FOD_RE(Abstract_Subfunction):
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    def __init__(self, mode="z-score"):
        self.mode = mode
        self.mean = self.std = self.min = self.max = None

        self.brain_img = self.brain_mask = None
        self.ref_img = self.ref_mask = None
        self.lesion_mask = None

        # self.mean = None
        # self.std = None
        # self.min = None
        # self.max = None

        # self.brain_img = None
        # self.brain_mask = None
        # self.ref_img = None
        # self.ref_mask = None
        # self.lesion_mask = None

    #---------------------------------------------#
    #                Preprocessing                #
    #---------------------------------------------#
    def preprocessing(self, sample, training=True, validation=True, opt=None):
        logger.info(f"[Preprocessing] Start with mode: {self.mode}")
        # Access input and mask
        brain_img = sample.img_data.squeeze().copy()
        brain_mask = sample.seg_data.squeeze()

        # # IF gt is loaded
        # if training or validation:
        #     ref_img = sample.gt_data.squeeze().copy()
        #     # ref mask: gt, healthy brain mask
        #     ref_mask = ref_img.copy()
        #     ref_mask[ref_mask != 0] = 1
        if training or validation:
            ref_img = sample.gt_data.squeeze().copy()
            ref_mask = (ref_img != 0).astype(np.uint8)

        if self.mode == "z-score":
            self._z_score(brain_img, brain_mask, ref_img, ref_mask)

        elif self.mode == 'z-score_v2':
            self._z_score_v2(brain_img, brain_mask, ref_img, training, opt)
            brain_mask = np.expand_dims(brain_mask, 3).repeat(45, axis=3)

        elif self.mode == 'z-score_v3':
            self._z_score_v3(brain_img, ref_img)
            brain_mask = np.expand_dims(brain_mask, 3).repeat(45, axis=3)

        elif self.mode == 'z-score_v4':
            self._z_score_v4(brain_img, brain_mask, ref_img, training)
            brain_mask = np.expand_dims(brain_mask, 3).repeat(45, axis=3)

        elif self.mode == "minmax":
            self._minmax(brain_img, brain_mask, ref_img, ref_mask)

        elif self.mode == "grayscale":
            self._grayscale(ref_img, ref_mask, brain_img, brain_mask)

        elif self.mode == "notnorm":
            logger.info(f"[notnorm] No normalization applied")

        else:
            logger.error(f"[error] Unknown normalization mode: {self.mode}")
            raise NameError("Unknown normalization mode")

        # Update the sample with the normalized image
        sample.img_data = brain_img
        sample.seg_data = brain_mask
        self.brain_img, self.brain_mask = brain_img, brain_mask

        if training or validation:
            sample.gt_data = ref_img
            self.ref_img, self.ref_mask = ref_img, ref_mask

        self.sample = sample
        logger.info(f"[Preprocessing] Finished")
    
    #---------------------------------------------#
    #               Postprocessing                #
    #---------------------------------------------#
    def postprocessing(self, prediction):
        logger.info(f"[Postprocessing] No postprocessing performed")
        return prediction
    
    # ----------------------------------#
    #      Internal normalization       #
    # ----------------------------------#
    def _z_score(self, brain_img, brain_mask, ref_img, ref_mask):
        mean = np.mean(brain_img[brain_mask == 1])
        std = np.std(brain_img[brain_mask == 1])
        self.mean, self.std = mean, std
        logger.info(f"[z-score] Mean: {mean:.4f}, Std: {std:.4f}")

        brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - mean) / std
        brain_img[brain_mask == 0] = 0
        ref_img[ref_mask == 1] = (ref_img[ref_mask == 1] - mean) / std
        ref_img[ref_mask == 0] = 0

    def _z_score_v2(self, brain_img, brain_mask, ref_img, training, opt):
        if training:
            self.fodlr_mean = np.mean(brain_img[brain_mask == 1], axis=0)
            self.fodlr_std = np.std(brain_img[brain_mask == 1], axis=0)
            self.fodgt_mean = np.mean(ref_img[brain_mask == 1], axis=0)
            self.fodgt_std = np.std(ref_img[brain_mask == 1], axis=0)
            logger.info(f"[z-score_v2] Computed mean/std for LR and GT (training)")
            for attr in ['fodlr_mean', 'fodlr_std', 'fodgt_mean', 'fodgt_std']:
                setattr(self, attr, getattr(self, attr).reshape(1, 1, 1, -1).astype(np.float32))
        else:
            path = os.path.join(opt.checkpoints_dir, opt.name)
            self.fodgt_mean = np.load(os.path.join(path, 'z-score_v2_gt_mean.npy'))
            self.fodgt_std = np.load(os.path.join(path, 'z-score_v2_gt_std.npy'))
            self.fodlr_mean = np.load(os.path.join(path, 'z-score_v2_lr_mean.npy'))
            self.fodlr_std = np.load(os.path.join(path, 'z-score_v2_lr_std.npy'))
            logger.info(f"[z-score_v2] Loaded stats from checkpoint")

        self.mean, self.std = self.fodgt_mean, self.fodgt_std
        brain_img[:] = (brain_img - self.fodlr_mean) / self.fodlr_std
        ref_img[:] = (ref_img - self.fodgt_mean) / self.fodgt_std

    def _z_score_v3(self, brain_img, ref_img):
        self.fodlr_mean = np.mean(brain_img, axis=(0,1,2)).reshape(1,1,1,-1)
        self.fodlr_std = np.std(brain_img, axis=(0,1,2)).reshape(1,1,1,-1)
        self.fodgt_mean = np.mean(ref_img, axis=(0,1,2)).reshape(1,1,1,-1)
        self.fodgt_std = np.std(ref_img, axis=(0,1,2)).reshape(1,1,1,-1)
        logger.info(f"[z-score_v3] Computed mean/std over all voxels")

        self.mean, self.std = self.fodgt_mean, self.fodgt_std
        brain_img[:] = (brain_img - self.fodlr_mean) / self.fodlr_std
        ref_img[:] = (ref_img - self.fodgt_mean) / self.fodgt_std

    def _z_score_v4(self, brain_img, brain_mask, ref_img, training):
        mean = np.mean(brain_img[brain_mask == 1], axis=0)
        std = np.std(brain_img[brain_mask == 1], axis=0)
        logger.info(f"[z-score_v4] Computed brain region stats ({'training' if training else 'validation'})")

        self.fodlr_mean = self.fodgt_mean = mean.reshape(1,1,1,-1).astype(np.float32)
        self.fodlr_std = self.fodgt_std = std.reshape(1,1,1,-1).astype(np.float32)
        self.mean, self.std = self.fodgt_mean, self.fodgt_std

        brain_img[:] = (brain_img - self.fodlr_mean) / self.fodlr_std
        ref_img[:] = (ref_img - self.fodgt_mean) / self.fodgt_std

    def _minmax(self, brain_img, brain_mask, ref_img, ref_mask):
        min_val = np.min(brain_img[brain_mask == 1])
        max_val = np.max(brain_img[brain_mask == 1])
        val_range = max_val - min_val
        self.min, self.max = min_val, max_val
        logger.info(f"[minmax] Min: {min_val:.4f}, Max: {max_val:.4f}")

        brain_img[brain_mask == 1] = (brain_img[brain_mask == 1] - min_val) / val_range
        brain_img[brain_mask == 0] = 0
        ref_img[ref_mask == 1] = (ref_img[ref_mask == 1] - min_val) / val_range
        ref_img[ref_mask == 0] = 0

    def _grayscale(self, ref_img, ref_mask, brain_img, brain_mask):
        max_val = np.max(brain_img[brain_mask == 1])
        min_val = np.min(brain_img[brain_mask == 1])
        logger.info(f"[grayscale] Min: {min_val:.4f}, Max: {max_val:.4f}")

        scaled = (ref_img[ref_mask == 1] - min_val) / (max_val - min_val)
        ref_img[ref_mask == 1] = scaled
        ref_img[ref_mask == 0] = 0
