#==============================================================================#
#  Project:     FOD Resolution Enhancement Framework                           #
#  File:        fod_re_io.py                                                   #
#  Authors:     Xinyi Wang                                                     #
#  License:     GNU General Public License v3.0                                #
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
from utils.mrtrix import load_mrtrix, save_mrtrix
from utils.logger import setup_logger

logger = setup_logger()

#-----------------------------------------------------#
#                 NIfTI I/O Interface                 #
#-----------------------------------------------------#
""" Data I/O Interface for NIfTI files. The Neuroimaging Informatics Technology Initiative file format
    is designed to contain brain images from e.g. magnetic resonance tomography. Nevertheless, it is
    currently broadly used for any 3D medical image data.

Code source heavily modified from the Kidney Tumor Segmentation Challenge 2019 git repository:
https://github.com/neheller/kits19
"""
class FOD_RE_interface(Abstract_IO):
    """
    A unified interface for loading and saving neuroimaging data in NIfTI or MRtrix format.
    Used for handling 3D/4D volumes for training and inference workflows.
    """
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
    def initialize(self, input_path):
        """
        Identify valid samples in the input directory.

        Args:
            input_path (str): Directory path to scan.

        Returns:
            list: Valid sample identifiers.
        """
        if not os.path.isdir(input_path):
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        sample_list = [
            fname[:-len(self.suffix)]
            for fname in os.listdir(input_path)
            if fname.endswith(self.suffix)
        ]

        if self.pattern:
            sample_list = [s for s in sample_list if re.match(self.pattern, s)]

        return sample_list
        
    #---------------------------------------------#
    #                  load_image                 #
    #---------------------------------------------#
    def load_image(self, index, input_path):
        """ Load input image volume by index from given path. """
        return self._load_volume(index, input_path, self.suffix, cache_affine=True)

    def load_gt(self, index, gt_path):
        """ Load ground truth volume by index. """
        return self._load_volume(index, gt_path, self.gt_suffix, cache_affine=True)

    def _load_volume(self, index, base_path, file_suffix, cache_affine=False):
        """
        Generalized volume loader that supports .nii, .nii.gz, .mif, .mif.gz formats.
        """
        path = os.path.join(base_path, index + file_suffix)
        logger.info(f"[Loading] Volume: {path}")
        # print(f"[Loading] {path}")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Volume not found: {path}")

        if path.endswith(('.mif.gz', '.mif')):
            mif = load_mrtrix(path)
            return mif.data, mif

        if path.endswith(('.nii.gz', '.nii')):
            vol = nib.load(path)
            if cache_affine:
                self.cache[index] = vol.affine
            return vol.get_fdata(), vol.affine

        raise ValueError(f"Unsupported file format: {path}")
    
    #---------------------------------------------#
    #              Load NIFTI mask                #
    #---------------------------------------------#
    def load_segmentation(self, index, mask_path):
        """ Load segmentation mask by index. """
        pass
        # return self._load_nifti(os.path.join(mask_path, f"{index}_fixel_mask.nii.gz"))

    def load_fixelmask(self, index, mask_path):
        """ Load fixel mask stored under subdirectory. """
        path = os.path.join(mask_path, str(index), "fixel_mask.nii.gz")
        return self._load_nifti(path, dtype=np.float32)

    def load_brainmask(self, index, mask_path):
        """ Load external brain mask by index. """
        path = os.path.join(mask_path, f"{index}_brain_mask.nii.gz")
        return self._load_nifti(path, dtype=np.float32)

    def _load_nifti(self, path, dtype=None):
        """ Load a NIfTI file and return data and affine. Optionally cast dtype. """
        # print(f"[Loading] {path}")
        logger.info(f"[Loading] NIfTI: {path}")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"NIfTI file not found: {path}")
        vol = nib.load(path)
        data = vol.get_fdata()
        if dtype is not None:
            data = data.astype(dtype)
        return data, vol.affine

    #---------------------------------------------#
    #               load_prediction               #
    #---------------------------------------------#
    def load_prediction(self, index, output_path):
        """ Load predicted output volume. """
        path = os.path.join(output_path, f"{index}.nii.gz")
        logger.info(f"[Loading] Prediction: {path}")
        # print(f"[Loading] {path}")

        if not os.path.isfile(path):
            raise FileNotFoundError(f"Prediction file not found: {path}")
        return nib.load(path).get_fdata()

    #---------------------------------------------#
    #                 load_details                #
    #---------------------------------------------#
    # Parse slice thickness
    def load_details(self, i):
        """ Extract voxel spacing from cached affine matrix. """
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

    #---------------------------------------------#
    #               save_prediction               #
    #---------------------------------------------#
    def save_prediction(self, pred, index, output_path, info=None):
        """
        Save a prediction volume in either NIfTI or MRtrix format based on file suffix.
        """
        os.makedirs(output_path, exist_ok=True)
        out_path = os.path.join(output_path, f"{index}.{self._get_extension()}")
        logger.info(f"[Saving] Prediction to: {out_path}")
        # print(f"[Saving] {out_path}")

        if self.suffix.endswith(('.mif.gz', '.mif')):
            if info is not None:
                info.data = pred
                save_mrtrix(out_path, info)
            else:
                raise ValueError("Missing MRtrix header information for saving.")

        elif self.suffix.endswith(('.nii.gz', '.nii')):
            if info is not None:
                nifti = nib.Nifti1Image(pred, affine=info)
                nib.save(nifti, out_path)
            else:
                raise ValueError("Missing affine matrix for saving NIfTI.")

        else:
            raise ValueError("Unsupported file format for saving.")
    
    def _get_extension(self):
        """ Infer file extension from suffix for saving predictions. """
        if self.suffix.endswith('.nii.gz'):
            return 'nii.gz'
        elif self.suffix.endswith('.nii'):
            return 'nii'
        elif self.suffix.endswith('.mif.gz'):
            return 'mif.gz'
        elif self.suffix.endswith('.mif'):
            return 'mif'
        else:
            raise ValueError(f"Unknown suffix format: {self.suffix}")
