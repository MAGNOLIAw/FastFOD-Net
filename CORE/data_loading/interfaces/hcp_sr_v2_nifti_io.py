#==============================================================================#
#  Author:       Dominik Müller, Xinyi Wang                                             #
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
import os
import nibabel as nib
import re
import numpy as np
import warnings
# Internal libraries/scripts
from data_loading.interfaces.abstract_io import Abstract_IO
from tqdm import tqdm
from utils.mrtrix import load_mrtrix, save_mrtrix, Image

#-----------------------------------------------------#
#                 NIfTI I/O Interface                 #
#-----------------------------------------------------#
""" Data I/O Interface for NIfTI files. The Neuroimaging Informatics Technology Initiative file format
    is designed to contain brain images from e.g. magnetic resonance tomography. Nevertheless, it is
    currently broadly used for any 3D medical image data.

Code source heavily modified from the Kidney Tumor Segmentation Challenge 2019 git repository:
https://github.com/neheller/kits19
"""
class HCP_SR_v2_NIFTI_interface(Abstract_IO):
    # Class variable initialization
    def __init__(self, channels=1, classes=2, three_dim=True, pattern='^\d{3}$',
                 suffix=None, gt_suffix=None):
        self.channels = channels
        self.classes = classes
        self.three_dim = three_dim
        self.pattern = pattern
        self.suffix = suffix
        self.gt_suffix = gt_suffix
        self.cache = dict()

    #---------------------------------------------#
    #                  initialize                 #
    #---------------------------------------------#
    # Initialize the interface and return number of samples
    def initialize(self, input_path):
        # Resolve location where imaging data should be living
        if not os.path.exists(input_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(input_path))
            )

        # Identify samples
        brain_name_list = os.listdir(os.path.join(input_path))
        sample_list = []
        for brain_name in tqdm(brain_name_list):
            # Skip if file does not end with nii.gz
            if not brain_name.endswith(self.suffix): continue
            sample_list.append(brain_name[:-len(self.suffix)])

        # IF pattern provided: Remove every file which does not match
        if self.pattern != None and isinstance(self.pattern, str):
            for i in reversed(range(0, len(sample_list))):
                if not re.match(self.pattern, sample_list[i]):
                    del sample_list[i]
        # print('sample list', sample_list)
        # Return sample list
        return sample_list

    # #---------------------------------------------#
    # #                  load_image                 #
    # #---------------------------------------------#
    # # Read a volume NIFTI file from the data directory
    # def load_image(self, index, input_path):
    #     print(index, input_path, self.suffix)
    #     path = os.path.join(input_path, index + self.suffix)
    #     if path.endswith('.mif.gz') or path.endswith('.mif'):
    #         image = load_mrtrix(path).data
    #         affine = None
    #     elif path.endswith('.nii.gz') or path.endswith('.nii'):
    #         vol = nib.load(path)
    #         affine = vol.affine
    #         # Transform NIFTI object to numpy array
    #         image = vol.get_data()
    #         # Save spacing in cache
    #         self.cache[index] = vol.affine
    #     else:
    #         image = None
    #         affine = None
    #         raise IOError('file extension not supported: ' + str(path))
    #
    #     # Make sure that the image file exists in the data set directory
    #     if not os.path.exists(path):
    #         raise ValueError(
    #             "Image could not be found \"{}\"".format(path)
    #         )
    #     print('loading from input_path: ', path)
    #     print('input vol', image.shape)
    #     # Return volume
    #     return image, affine

    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    # Read a volume NIFTI file from the data directory
    def load_image(self, index, input_path):
        print(index, input_path, self.suffix)
        path = os.path.join(input_path, index + self.suffix)
        if path.endswith('.mif.gz') or path.endswith('.mif'):
            mif = load_mrtrix(path)
            image = mif.data
            info = mif
        elif path.endswith('.nii.gz') or path.endswith('.nii'):
            vol = nib.load(path)
            info = vol.affine
            # Transform NIFTI object to numpy array
            image = vol.get_data()
            # Save spacing in cache
            self.cache[index] = vol.affine
        else:
            image = None
            info = None
            raise IOError('file extension not supported: ' + str(path))

        # Make sure that the image file exists in the data set directory
        if not os.path.exists(path):
            raise ValueError(
                "Image could not be found \"{}\"".format(path)
            )
        print('loading from input_path: ', path)
        print('input vol', image.shape)
        # Return volume
        return image, info

    # #---------------------------------------------#
    # #              load_gt             #
    # #---------------------------------------------#
    # # Read a volume NIFTI file from the data directory
    # def load_gt(self, index, gt_path):
    #     # Make sure that the image file exists in the data set directory
    #     print(index, gt_path)
    #     path = os.path.join(gt_path, index + self.gt_suffix)
    #     if path.endswith('.mif.gz') or path.endswith('.mif'):
    #         image = load_mrtrix(path).data
    #         affine = None
    #     elif path.endswith('.nii.gz') or path.endswith('.nii'):
    #         vol = nib.load(path)
    #         affine = vol.affine
    #         # Transform NIFTI object to numpy array
    #         image = vol.get_data()
    #         # Save spacing in cache
    #         self.cache[index] = vol.affine
    #     else:
    #         image = None
    #         affine = None
    #         raise IOError('file extension not supported: ' + str(path))
    #
    #     print('loading from gt_path: ', path)
    #     print('gt vol', image.shape)
    #     if not os.path.exists(gt_path):
    #         raise ValueError(
    #             "Gt could not be found \"{}\"".format(gt_path)
    #         )
    #     # Return volume
    #     return image, affine

    #---------------------------------------------#
    #              load_gt             #
    #---------------------------------------------#
    # Read a volume NIFTI file from the data directory
    def load_gt(self, index, gt_path):
        # Make sure that the image file exists in the data set directory
        print(index, gt_path)
        path = os.path.join(gt_path, index + self.gt_suffix)
        if path.endswith('.mif.gz') or path.endswith('.mif'):
            image = load_mrtrix(path).data
            info = None
        elif path.endswith('.nii.gz') or path.endswith('.nii'):
            vol = nib.load(path)
            info = vol.affine
            # Transform NIFTI object to numpy array
            image = vol.get_data()
            # Save spacing in cache
            self.cache[index] = vol.affine
        else:
            image = None
            info = None
            raise IOError('file extension not supported: ' + str(path))

        print('loading from gt_path: ', path)
        print('gt vol', image.shape)
        if not os.path.exists(gt_path):
            raise ValueError(
                "Gt could not be found \"{}\"".format(gt_path)
            )
        # Return volume
        return image, info

    #---------------------------------------------#
    #              load_segmentation              #
    #---------------------------------------------#
    # Read a segmentation NIFTI file from the data directory
    def load_segmentation(self, index, mask_path=None):
        # Make sure that the segmentation file exists in the data set directory
        seg_path = os.path.join(mask_path)
        print('loading from mask_path: ', mask_path, str(index) + "_fixel_mask.nii.gz")
        if not os.path.exists(seg_path):
            raise ValueError(
                "Mask could not be found \"{}\"".format(seg_path)
            )
        # Load segmentation from NIFTI file
        seg = nib.load(os.path.join(mask_path, str(index) + "_fixel_mask.nii.gz"))
        # Transform NIFTI object to numpy array
        seg_data = seg.get_fdata()
        # Return segmentation
        seg_aff = seg.affine

        return seg_data, seg_aff

    def load_fixelmask(self, index, mask_path=None):
        # Make sure that the segmentation file exists in the data set directory
        # path = '/home/mariano/HCP_fixel_masks/'
        path = mask_path
        print('loading from mask_path: ', path, str(index), "fixel_mask.nii.gz")
        if not os.path.exists(path):
            raise ValueError(
                "Mask could not be found \"{}\"".format(path)
            )
        # Load segmentation from NIFTI file
        seg = nib.load(os.path.join(path, str(index), "fixel_mask.nii.gz"))
        # Transform NIFTI object to numpy array
        seg_data = np.float32(seg.get_fdata())
        # seg_data = seg.get_fdata()
        # Return segmentation
        seg_aff = seg.affine
        return seg_data, seg_aff

    def load_brainmask(self, index, mask_path=None):
        # Make sure that the segmentation file exists in the data set directory
        # path = '/home/mariano/HCP_fixel_masks/'
        path = mask_path
        print('loading from mask_path: ', path, str(index) + "_brain_mask.nii.gz")
        if not os.path.exists(path):
            raise ValueError(
                "Mask could not be found \"{}\"".format(path)
            )
        # Load segmentation from NIFTI file
        seg = nib.load(os.path.join(path, str(index) + "_brain_mask.nii.gz"))
        # Transform NIFTI object to numpy array
        seg_data = np.float32(seg.get_fdata())
        # seg_data = seg.get_fdata()
        # Return segmentation
        seg_aff = seg.affine
        return seg_data, seg_aff

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    # Read a prediction NIFTI file from the MIScnn output directory
    def load_prediction(self, index, output_path):
        # Resolve location where data should be living
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(str(output_path))
            )
        # Parse the provided index to the prediction file name
        pred_file = str(index) + ".nii.gz"
        pred_path = os.path.join(output_path, pred_file)
        print('loading from pred_path: ', pred_path)
        # Make sure that prediction file exists under the prediction directory
        if not os.path.exists(pred_path):
            raise ValueError(
                "Prediction could not be found \"{}\"".format(pred_path)
            )
        # Load prediction from NIFTI file
        pred = nib.load(pred_path)
        # Transform NIFTI object to numpy array
        pred_data = pred.get_data()
        # Return prediction
        return pred_data

    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    # Parse slice thickness
    def load_details(self, i):
        # Parse voxel spacing from affinity matrix of NIfTI
        spacing_matrix = self.cache[i][:3,:3]
        # Identify correct spacing diagonal
        diagonal_negative = np.diag(spacing_matrix)
        diagonal_positive = np.diag(spacing_matrix[::-1,:])
        if np.count_nonzero(diagonal_negative) != 1:
            spacing = diagonal_negative
        elif np.count_nonzero(diagonal_positive) != 1:
            spacing = diagonal_positive
        else:
            warnings.warn("Affinity matrix of NIfTI volume can not be parsed.")
        # Calculate absolute values for voxel spacing
        spacing = np.absolute(spacing)
        # Delete cached spacing
        del self.cache[i]
        # Return detail dictionary
        return {"spacing":spacing}

    # #---------------------------------------------#
    # #               save_prediction               #
    # #---------------------------------------------#
    # # Write a segmentation prediction into in the NIFTI file format
    # def save_prediction(self, pred, index, output_path, aff=None):
    #     # Resolve location where data should be written
    #     if not os.path.exists(output_path):
    #         raise IOError(
    #             "Data path, {}, could not be resolved".format(output_path)
    #         )
    #
    #     if self.suffix.endswith('.mif.gz') or self.suffix.endswith('.mif'):
    #         img = Image(pred)
    #         pred_file = str(index) + ".mif.gz"
    #         print('pred_file', pred_file)
    #         save_mrtrix(os.path.join(output_path, pred_file), img)
    #     elif self.suffix.endswith('.nii.gz') or self.suffix.endswith('.nii'):
    #         # Convert numpy array to NIFTI
    #         nifti = nib.Nifti1Image(pred, affine=aff)
    #         # Save segmentation to disk
    #         pred_file = str(index) + ".nii.gz"
    #         nib.save(nifti, os.path.join(output_path, pred_file))

    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    # Write a segmentation prediction into in the NIFTI file format
    def save_prediction(self, pred, index, output_path, info=None):
        # Resolve location where data should be written
        if not os.path.exists(output_path):
            raise IOError(
                "Data path, {}, could not be resolved".format(output_path)
            )

        if self.suffix.endswith('.mif.gz') or self.suffix.endswith('.mif'):
            # img = Image(pred)
            info.data = pred
            pred_file = str(index) + ".mif.gz"
            save_mrtrix(os.path.join(output_path, pred_file), info)
        elif self.suffix.endswith('.nii.gz') or self.suffix.endswith('.nii'):
            # Convert numpy array to NIFTI
            nifti = nib.Nifti1Image(pred, affine=info)
            # Save segmentation to disk
            pred_file = str(index) + ".nii.gz"
            nib.save(nifti, os.path.join(output_path, pred_file))
