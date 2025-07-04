#==============================================================================#  
#  Authors:     Xinyi Wang                                                     #
#  License:     GNU GPL v3.0                                                   #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import pickle

# Project-specific modules
import data_loading.fod_re_sample as FOD_sample
from utils.logger import setup_logger

logger = setup_logger()



#-----------------------------------------------------#
#                    Data IO class                    #
#-----------------------------------------------------#
# Class to handle all input and output functionality
class FOD_RE_Data_IO:
    """
    A general I/O handler for loading, saving, and backing up diffusion MRI samples.

    This class coordinates input/output interfaces and allows for modular handling
    of image data, ground truth, predictions, and segmentation masks.
    """

    def __init__(self, interface, input_path, output_path="predictions",
                 batch_path="batches", delete_batchDir=True, mask_path=None,
                 gt_path=None, sample_list=None):
        """
        Initialize the Data IO module.

        Args:
            interface (object): I/O interface class (inherits from abstract_io).
            input_path (str): Directory path to the raw input image data.
            output_path (str): Directory to save predictions.
            batch_path (str): Directory for temporarily backed-up samples.
            delete_batchDir (bool): Whether to remove all batch files after use.
            mask_path (str): Optional path to precomputed brain masks.
            gt_path (str): Directory path to ground truth data.
            sample_list (list): Optional list of subject/sample identifiers.
        """       
        self.interface = interface
        self.input_path = input_path
        self.output_path = output_path
        self.batch_path = batch_path
        self.delete_batchDir = delete_batchDir
        self.mask_path = mask_path
        self.gt_path = gt_path
        self.seed = 6666

        # Initialize Data I/O interface
        if sample_list is None:
            self.indices_list = interface.initialize(input_path)
        else:
            self.indices_list = sample_list
        # self.indices_list.sort()

    #---------------------------------------------#
    #                Sample Loader                #
    #---------------------------------------------#
    # Load a sample from the data set
    def sample_loader(self, index, load_seg=True, load_pred=False, backup=False,
                      load_gt=True, load_brainmask=False, load_fixelmask=False):
        """
        Load a single sample for inference or evaluation.

        Args:
            index (str or int): Sample identifier.
            load_seg (bool): Whether to generate brain mask from input image.
            load_pred (bool): Whether to load prediction from disk.
            backup (bool): Whether to load sample from a saved pickle file.
            load_gt (bool): Whether to load the ground truth FOD.
            load_brainmask (bool): Whether to load externally provided brain mask.
            load_fixelmask (bool): Whether to load fixel-based mask instead.

        Returns:
            FOD_sample.Sample: An assembled sample object containing image, mask, GT, etc.
        """

        # If sample is a backup, load it from pickle
        if backup: 
            return self.load_sample_pickle(index)

        # Load the image with the I/O interface
        image, info = self.interface.load_image(index, self.input_path)
        # Create a Sample object
        sample = FOD_sample.Sample(index, image, self.interface.channels, self.interface.classes, info)
        # Load ground truth if specified
        if load_gt:
            gt, info = self.interface.load_gt(index, self.gt_path)
            sample.add_gt(gt)

        # Load segmentation
        if load_seg:
            # use non zeros of input as mask
            brain_mask = (image[:, :, :, 0] != 0).astype(image.dtype)
            sample.add_segmentation(brain_mask)
        elif load_brainmask or load_fixelmask:
            mask, mask_info = self.interface.load_brainmask(index, self.mask_path)
            sample.add_segmentation(mask, mask_info)
        else:
            logger.warning("No valid segmentation loaded.")
            # print("[Warning] No valid segmentation loaded.")

        # Load prediction if required
        if load_pred:
            prediction = self.interface.load_prediction(index, self.output_path)
            sample.add_prediction(prediction)

        # Load additional details if available
        # if info is not None:
        #     details = self.interface.load_details(index)
        #     sample.add_details(details)
        # print('sample.details', sample.details)

        return sample

    #---------------------------------------------#
    #              Prediction Backup              #
    #---------------------------------------------#
    def save_prediction(self, pred, index, info=None, path=None):
        """
        Save a prediction result to the specified output directory.

        Args:
            pred (ndarray): The predicted image or tensor.
            index (str or int): Identifier of the sample.
            info (dict, optional): Metadata or header info.
            path (str, optional): Output directory. If None, use default output path.
        """
        save_path = path if path is not None else self.output_path

        os.makedirs(save_path, exist_ok=True)
        self.interface.save_prediction(pred, index, save_path, info)

    #---------------------------------------------#
    #                Sample Backup                #
    #---------------------------------------------#
    def backup_sample(self, sample):
        """
        Backup a sample object to disk as a pickle file.

        Args:
            sample (FOD_sample.Sample): The sample object to be backed up.
        """
        os.makedirs(self.batch_path, exist_ok=True)
        sample_filename = f"{self.seed}.{sample.index}.pickle"
        sample_path = os.path.join(self.batch_path, sample_filename)

        if not os.path.exists(sample_path):
            with open(sample_path, 'wb') as f:
                pickle.dump(sample, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"[Backup] Sample saved at: {sample_path}")
            # print(f"[Info] Sample backed up at: {sample_path}")
        else:
            logger.info(f"[Backup] Sample already exists at: {sample_path}")
            # print(f"[Info] Sample already exists at: {sample_path}")

    def run_backup(self, indices_list):
        """
        Run batch backup for a list of indices.

        Args:
            indices_list (list): List of sample identifiers to back up.
            training (bool): Flag indicating whether this is part of training (not used).
        """
        for index in indices_list:
            if index == '.DS_Store':
                continue
            sample = self.sample_loader(index, load_seg=True, load_pred=False, load_gt=False)
            self.backup_sample(sample)
            logger.info(f"[Backup] Case processed: {index}")
            # print(f"[Backup] Case saved: {index}")
        logger.info("All samples backed up successfully.")

    def load_sample_pickle(self, index):
        """
        Load a previously backed-up sample from disk.

        Args:
            index (str or int): Identifier of the sample to load.

        Returns:
            FOD_sample.Sample: The deserialized sample object.
        """
        sample_filename = f"{self.seed}.{index}.pickle"
        sample_path = os.path.join(self.batch_path, sample_filename)

        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample pickle not found at {sample_path}")

        with open(sample_path, 'rb') as f:
            sample = pickle.load(f)

        return sample

    #-----------------------------------------------------#
    #              Accessor for Sample Indices            #
    #-----------------------------------------------------#
    def get_indiceslist(self):
        """
        Get a copy of the current list of sample indices.

        Returns:
            list: A copy of the indices list.
        """
        return self.indices_list.copy()
