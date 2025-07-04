#==============================================================================#
#  Project:     Diffusion MRI Patch-Based Processing                           #
#  File:        fod_re_dataset.py                                              #
#  Authors:     Xinyi Wang                                                     #
#  License:     GNU GPL v3.0                                                   #
#==============================================================================#
#  Description:                                                                #
#    PyTorch Dataset class for patch-wise training and inference of FOD        #
#    volumes with optional bounding-box cropping, normalization, and           #
#    caching. Handles both 3D and 4D volumes across different modes.           #
#==============================================================================#

import os
import torch
from data.base_dataset import BaseDataset
import numpy as np
import nibabel as nib
import pickle

from utils.patch_operations import (
    find_bounding_box, slice_matrix, concat_matrices,
    pad_patch, crop_patch
)
from processing.subfunctions import Normalization_FOD_RE
from utils.logger import setup_logger

logger = setup_logger()

# def log(msg, level="INFO"):
#     print(f"[{level}] {msg}")

class FODREDataset(BaseDataset):
    """
    A dataset class for brain image processing.

    Supports training, validation, and test phases with patch-wise processing,
    optional bounding box cropping, and normalization for FOD data.
    """

    def __init__(self, opt):
        """
        Initialize this dataset class.
        """
        BaseDataset.__init__(self, opt)
        self.sub_list = os.listdir(opt.dataroot)
        self.opt = opt

        self.input_nc = self.opt.input_nc
        self.output_nc = self.opt.output_nc
        self.bounding_box = self.opt.bounding_box

        self.patch_shape = (64, 64, 64)
        self.patchwise_overlap = (32, 32, 32)
        self.patchwise_skip_blanks = opt.patchwise_skip_blanks
        self.prepare_batches = False
        self.analysis = 'patchwise-grid'

        shuffle_batches = opt.shuffle_batches

        # Set up training or test or validation
        if opt.phase == 'train':
            self.set_up(opt.sample_list, training=True, validation=False, shuffle=shuffle_batches)
        elif opt.phase in ['test', 'validation']:
            sample = [opt.sample_list[opt.eval_sample]]
            validation = opt.phase == 'validation'
            self.set_up(sample, training=False, validation=validation, shuffle=shuffle_batches)
        else:
            raise ValueError(f"Invalid phase: {opt.phase}")

    def set_up(self, indices_list, training=True, validation=False, shuffle=False):
        """
        Load and preprocess samples for the dataset.

        Args:
            indices_list (list): List of subject/sample indices to load.
            training (bool): Whether it's training mode.
            validation (bool): Whether it's validation mode.
            shuffle (bool): Whether to shuffle sample list.
        """
        self.coord_queue = [] # a list of dicts of coordinate info
        self.samples = {}
        self.cache = {}
        if self.bounding_box: self.samples_bounding_box = {}

        # For normalization
        lr_mean, lr_std, gt_mean, gt_std = [], [], [], []

        # Iterate over all samples
        # logger.info('Load data and preprocessing')
        for i, index in enumerate(indices_list):
            # logger.info(f"Loading sample: {index}")

            # Load sample
            # if not self.prepare_batches:
            #     if not training and not validation:
            #         # test: load without gt
            #         sample = self.opt.data_io.sample_loader(index, load_seg=True, load_pred=False, backup=False, load_gt=False)
            #     else:
            #         sample = self.opt.data_io.sample_loader(index, load_seg=True, load_pred=False, backup=False, load_gt=True)
            # # Load sample from backup
            # else:
            #     sample = self.opt.data_io.sample_loader(index, backup=True)
            sample = self.opt.data_io.sample_loader(
                index,
                load_seg=True,
                load_pred=False,
                backup=self.prepare_batches,
                load_gt=(training or validation)
            )

            # Get bounding boxes of brain
            if self.bounding_box:
                logger.info(f"Performing bounding box cropping for sample: {index}")
                if not training:
                    self.cache[f"shape_before_bounding_{index}"] = sample.img_data.shape
                    logger.debug(f"Original shape before bounding for sample {index}: {self.cache[f'shape_before_bounding_{index}']}")
                bb = find_bounding_box(sample.img_data)
                sample.img_data = sample.img_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                sample.seg_data = sample.seg_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                if self.opt.phase != 'test':
                    sample.gt_data = sample.gt_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                self.samples_bounding_box[index] = bb
                logger.debug(f"Shapes after cropping → image: {sample.img_data.shape}, seg: {sample.seg_data.shape}, gt: {sample.gt_data.shape}")

            elif self.opt.bboxroot:
                logger.info(f"Loading bounding box from file for sample: {index}")
                with open(self.opt.bboxroot, 'rb') as handle:
                    bb = pickle.load(handle)[str(index)]['bbox']
                sample.img_data = sample.img_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                sample.seg_data = sample.seg_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                if self.opt.phase != 'test':
                    sample.gt_data = sample.gt_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                logger.debug(f"Shapes after loading bbox → image: {sample.img_data.shape}, seg: {sample.seg_data.shape}, gt: {sample.gt_data.shape}")

            if not training:
                self.cache[f"shape_{index}"] = sample.img_data.shape

            # Patch slicing before norm, return coordinates of patches
            logger.info(f"[PATCH] Slicing image into patches for sample: {index}")
            coords_img_data = self.analysis_patchwise_grid(sample, training, index)
            self.coord_queue.extend(coords_img_data)
            logger.info(f"[PATCH] Sample {index} | Original shape: {sample.img_data.shape} | Patches: {len(coords_img_data)} | Total so far: {len(self.coord_queue)}")

            # Normalization
            # logger.info(f"Starting normalization for sample: {index} (mode: {self.opt.normalization_mode})")
            sf_zscore = Normalization_FOD_RE(mode=self.opt.normalization_mode)
            # Assemble Subfunction classes into a list
            subfunctions = [sf_zscore]
            for sf in subfunctions:
                sf.preprocessing(sample, training=training, validation=validation, opt=self.opt)
                if 'z-score' in sf.mode:
                    gt_mean.append(sf.fodgt_mean)
                    gt_std.append(sf.fodgt_std)
                    lr_mean.append(sf.fodlr_mean)
                    lr_std.append(sf.fodlr_std)
            
            self.sf_zscore = sf_zscore
            self.subfunctions = subfunctions

            # Save the mean and std of all cases during training
            if training:
                checkpoint_path = os.path.join(self.opt.checkpoints_dir, self.opt.name)
                np.save(os.path.join(checkpoint_path, f'{sf.mode}_gt_mean.npy'), np.mean(gt_mean, axis=0), allow_pickle=True)
                np.save(os.path.join(checkpoint_path, f'{sf.mode}_gt_std.npy'), np.mean(gt_std, axis=0), allow_pickle=True)
                np.save(os.path.join(checkpoint_path, f'{sf.mode}_lr_mean.npy'), np.mean(lr_mean, axis=0), allow_pickle=True)
                np.save(os.path.join(checkpoint_path, f'{sf.mode}_lr_std.npy'), np.mean(lr_std, axis=0), allow_pickle=True)

            self.samples[index] = sample
        # logger.info("[Preprocessing] Done")   
        logger.info("-" * 55)

    #---------------------------------------------#
    #           Patch-wise grid Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_grid(self, sample, training, index=None):
        """
        Slice a 3D volume into overlapping patches.

        Args:
            sample: A loaded sample with image data.
            training (bool): Whether in training mode to filter background patches.
            index (optional): Sample index used for caching.

        Returns:
            np.ndarray: Array of patch coordinates.
        """
        # Slice image into patches
        patches_img, coords_img = slice_matrix(
            sample.img_data, self.patch_shape, self.patchwise_overlap,
            self.opt.data_io.interface.three_dim, index, save_coords=True
        )
        # Skip blank patches (only background)
        if training and self.patchwise_skip_blanks:
            for i in reversed(range(len(patches_img))):
                # IF patch DON'T contain any non background class -> remove it
                if np.sum(patches_img[i]) == 0:
                    del patches_img[i]
                    del coords_img[i]
        return np.stack(coords_img, axis=0)


    #---------------------------------------------#
    #          Prediction Postprocessing          #
    #---------------------------------------------#
    # Postprocess prediction data
    def postprocessing(self, sample, prediction, shape=None, coords=None):
        """
        Reconstruct prediction from patches and apply postprocessing steps.

        Args:
            sample: Sample identifier used for cache lookup.
            prediction: The predicted output patch-wise.
            shape: Optional original shape for reassembly.
            coords: Patch coordinates.

        Returns:
            np.ndarray: Final postprocessed prediction volume.
        """
        # Reassemble patches into original shape for patchwise analysis
        if self.analysis in ["patchwise-crop", "patchwise-grid"]:
            # Check if patch was padded
            slice_key = f"slicer_{sample}"
            if slice_key in self.cache:
                prediction = crop_patch(prediction, self.cache[slice_key])
                
            # Load cached shape & Concatenate patches into original shape
            shape = self.cache.get(f"shape_{sample}", shape)
            prediction = concat_matrices(
                patches=prediction, image_size=shape,
                window=self.patch_shape, overlap=self.patchwise_overlap,
                three_dim=self.opt.data_io.interface.three_dim, coords=coords
            )
        else: 
            # For full images remove the batch axis
            prediction = np.squeeze(prediction, axis=0)

        # Run Subfunction postprocessing on the prediction
        for sf in reversed(self.subfunctions):
            prediction = sf.postprocessing(prediction)
        
        if self.bounding_box:
            logger.debug(f"Reconstructing prediction to original shape: {shape}")
            # log(f"Reconstructing prediction to original shape: {shape}", level='DEBUG')
            original_shape = self.cache.pop(f"shape_before_bounding_{sample}")
            tmp = np.zeros(original_shape)
            bb = self.samples_bounding_box[sample]
            tmp[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] = prediction
            prediction = tmp

        return prediction

    # Return the next batch for associated index
    def __getitem__(self, idx):
        """
        Fetch a single sample from the dataset given a patch index.

        Args:
            idx (int): Index into the patch queue.

        Returns:
            dict: Dictionary containing 'brain', 'mask', and optionally 'gt' tensors.
        """

        # self.coords_batches = self.coord_queue
        # self.now_sample = self.samples[self.coords_batches[idx]['index']]

        coords = self.coord_queue[idx] 
        sample = self.samples[coords['index']]

        # x_start = self.coords_batches[idx]['x_start']
        # x_end = self.coords_batches[idx]['x_end']
        # y_start = self.coords_batches[idx]['y_start']
        # y_end = self.coords_batches[idx]['y_end']
        # z_start = self.coords_batches[idx]['z_start']
        # z_end = self.coords_batches[idx]['z_end']
        
        self.coord_queue[idx]['dataloader_idx'] = idx

        # x = torch.from_numpy(sample.img_data[x_start:x_end, y_start:y_end, z_start:z_end])
        # y = torch.from_numpy(sample.seg_data[x_start:x_end, y_start:y_end, z_start:z_end])

        x = torch.from_numpy(sample.img_data[coords['x_start']:coords['x_end'],
                                              coords['y_start']:coords['y_end'],
                                              coords['z_start']:coords['z_end']])
        y = torch.from_numpy(sample.seg_data[coords['x_start']:coords['x_end'],
                                              coords['y_start']:coords['y_end'],
                                              coords['z_start']:coords['z_end']])

        # w h d c
        # brain = x.clone()
        # mask = y.clone()
        # w h d c => c w h d
        brain = x.permute(3,0,1,2)
        mask = y.permute(3,0,1,2)

        if brain.shape[1:-1] != mask.shape[1:-1]:
            logger.warning(f"Shape mismatch → brain: {brain.shape}, mask: {mask.shape}")

        data = {'brain': brain, 'mask': mask}

        if self.opt.phase != 'test':
            gt = torch.from_numpy(sample.gt_data[coords['x_start']:coords['x_end'],
                                                 coords['y_start']:coords['y_end'],
                                                 coords['z_start']:coords['z_end']])
            data['gt'] = gt.permute(3, 0, 1, 2)
    
        return data

    # Return the number of batches for one epoch
    def __len__(self):
        return len(self.coord_queue)

    # At every epoch end: Shuffle batchPointer list and reset sample_list
    def on_epoch_end(self):
        if self.opt.shuffle_batches and self.opt.phase == 'train':
            if self.prepare_batches:
                np.random.shuffle(self.batchpointers)
            else:
                np.random.shuffle(self.sub_list)

    @staticmethod
    def modify_commandline_options(parser, isTrain):
        """
        Add any new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--test_fold', default='1', type=int, help='Test fold number')
        parser.add_argument('--shuffle_batches', default=True, type=bool, help='Shuffle batch order')
        parser.add_argument('--iterations', default=None, type=int, help='Number of iterations (batches) in a single epoch.')
        parser.add_argument('--patchwise_skip_blanks', default=False, type=bool, help='Skip blank patches')
        parser.add_argument('--gtroot', default=None, type=str, help='Path to ground truth data')
        parser.add_argument('--bounding_box', default=False, type=str, help='Enable bounding box cropping')
        parser.add_argument('--normalization_mode', default='z-score_v2', type=str, help='Normalization mode')

        return parser
