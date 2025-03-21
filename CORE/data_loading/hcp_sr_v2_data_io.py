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
import os
import pickle
# Internal libraries/scripts
import data_loading.hcp_fod_sample as MIScnn_sample

#-----------------------------------------------------#
#                    Data IO class                    #
#-----------------------------------------------------#
# Class to handle all input and output functionality
class HCP_SR_v2_Data_IO:
    # Class variables
    interface = None                    # Data I/O interface
    input_path = None                   # Path to input data directory
    output_path = None                  # Path to MIScnn prediction directory
    batch_path = None                   # Path to temporary batch storage directory
    indices_list = None                 # List of sample indices after data set initialization
    delete_batchDir = None              # Boolean for deletion of complete tmp batches directory
                                        # or just the batch data for the current seed
    # seed = random.randint(0,99999999)   # Random seed if running multiple MIScnn instances
    seed = 6666
    #---------------------------------------------#
    #                Initialization               #
    #---------------------------------------------#
    """ Initialization function for creating an object of the Data IO class.
    This class provides functionality for handling all input and output processes
    of the imaging data, as well as the temporary backup of batches to the disk.

    The user is only required to create an instance of the Data IO class with the desired specifications
    and IO interface for the correct format. It is possible to create a custom IO interface for handling
    special data structures or formats.

    Args:
        interface (io_interface):   A data IO interface which inherits the abstract_io class with the following methods:
                                    initialize, load_image, load_segmentation, load_prediction, save_prediction
        input_path (string):        Path to the input data directory, in which all imaging data have to be accessible.
        output_path (string):       Path to the output data directory, in which computed predictions will berun_evaluation.pyrun_evaluation.py stored. This directory
                                    will be created.
        batch_path (string):        Path to the batch data directory. This directory will be created and used for temporary files.
        delete_batchDir (boolean):  Boolean if the whole temporary batch directory for prepared batches should be deleted after
                                    model utilization. If false only the batches with the associated seed will be deleted.
                                    This parameter is important when running multiple instances of MIScnn.
    """
    def __init__(self, interface, input_path, output_path="predictions",
                 batch_path="batches", delete_batchDir=True, mask_path=None,
                 gt_path=None, sample_list=None):
        # Parse parameter
        self.interface = interface
        self.input_path = input_path
        self.output_path = output_path
        self.batch_path = batch_path
        self.delete_batchDir = delete_batchDir
        # Initialize Data I/O interface
        if sample_list == None:
            self.indices_list = interface.initialize(input_path)
        else:
            self.indices_list = sample_list
        # self.indices_list.sort()

        self.mask_path = mask_path
        self.gt_path = gt_path

    #---------------------------------------------#
    #                Sample Loader                #
    #---------------------------------------------#
    # Load a sample from the data set
    def sample_loader(self, index, load_seg=True, load_pred=False, backup=False,
                      load_gt=True, load_brainmask=False, load_fixelmask=False):
        # If sample is a backup -> load it from pickle
        if backup: return self.load_sample_pickle(index)
        # Load the image with the I/O interface
        image, info = self.interface.load_image(index, self.input_path)
        # Create a Sample object
        sample = MIScnn_sample.HCP_FOD_Sample(index, image, self.interface.channels,
                                              self.interface.classes, info)

        if load_gt:
            gt, info = self.interface.load_gt(index, self.gt_path)
            sample.add_gt(gt)

        # IF needed read the provided segmentation for current sample
        if load_seg:
            # use non zeros of input as mask
            brain_mask = image[:,:,:,0].copy()
            brain_mask[brain_mask != 0] = 1
            sample.add_segmentation(brain_mask)
            # print('sample.seg_data', sample.seg_data.shape)
        elif load_brainmask:
            # if load brainmask
            segmentation, info = self.interface.load_brainmask(index, self.mask_path)
            sample.add_segmentation(segmentation, info)
            # print('sample.seg_data', sample.seg_data.shape)
        elif load_fixelmask:
            # if load brainmask
            segmentation, info = self.interface.load_brainmask(index, self.mask_path)
            sample.add_segmentation(segmentation, info)
            # print('sample.seg_data', sample.seg_data.shape)
        else:
            print("No valid masks loaded!")

        # IF needed read the provided prediction for current sample
        if load_pred:
            prediction = self.interface.load_prediction(index, self.output_path)
            sample.add_prediction(prediction)
        # IF load NIFTI instead of mif
        if info is not None:
            # Add optional details to the sample object
            sample.add_details(self.interface.load_details(index))
        # Return sample object
        return sample

    #---------------------------------------------#
    #              Prediction Backup              #
    #---------------------------------------------#
    # Save a segmentation prediction
    def save_prediction(self, pred, index, info=None, path=None):
        if path is not None:
            # Create the output directory if not existent
            if not os.path.exists(path):
                os.mkdir(path)
            # Backup the prediction
            self.interface.save_prediction(pred, index, path, info)
        else:
            # Create the output directory if not existent
            if not os.path.exists(self.output_path):
                os.mkdir(self.output_path)
            # Backup the prediction
            self.interface.save_prediction(pred, index, self.output_path, info)

    #---------------------------------------------#
    #                Sample Backup                #
    #---------------------------------------------#
    # Backup samples for later access
    def backup_sample(self, sample):
        if not os.path.exists(self.batch_path) : os.mkdir(self.batch_path)
        sample_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                                   sample.index + ".pickle")
        if not os.path.exists(sample_path):
            with open(sample_path, 'wb') as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('Back up sample in:', sample_path)

    # backup samples as pickles
    def run_backup(self, indices_list, training=True):
        # Iterate over all samples
        for index in indices_list:
            if index != '.DS_Store':
                # Load sample
                sample = self.sample_loader(index, load_seg=True, load_pred=False, load_gt=False)
                # Backup sample as pickle to disk
                self.backup_sample(sample)
                print('backup case: ', index)
                # break
        print('-----------------------------------------------------------')

    # Load a backup sample from pickle
    def load_sample_pickle(self, index):
        sample_path = os.path.join(self.batch_path, str(self.seed) + "." + \
                                   index + ".pickle")
        # print(index, sample_path)
        with open(sample_path,'rb') as reader:
            sample = pickle.load(reader)
        # print('Load sample pickle from:', sample_path)
        return sample

    #---------------------------------------------#
    #               Variable Access               #
    #---------------------------------------------#
    def get_indiceslist(self):
        return self.indices_list.copy()
