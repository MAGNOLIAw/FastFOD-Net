import os
import torch
from data.base_dataset import BaseDataset
import numpy as np
import nibabel as nib

from utils.patch_operations import find_bounding_box, slice_matrix, concat_matrices, pad_patch, crop_patch, find_bounding_box
from processing.subfunctions import Normalization_HCP_SR_FOD

class FODv2RealtimeDataset(BaseDataset):
    """A dataset class for brain image dataset.

    It assumes that the directory '/path/to/data/train' contains brain image slices
    in torch.tensor format to speed up the I/O. Otherwise, you can load MRI brain images
    using nibabel package and preprocess the slices during loading period.
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
        # self.bounding_box = False
        self.bounding_box = self.opt.bounding_box

        # self.patch_shape=(112,112,64)
        # self.patchwise_overlap=(56,56,32)
        self.patch_shape=(64,64,64)
        self.patchwise_overlap=(32,32,32)
        # self.patch_shape = (112,112,64)
        # self.patchwise_overlap = (56,56,32)
        # self.patch_shape=(96,96,96)
        # self.patchwise_overlap=(48,48,48)
        # self.patchwise_overlap = (0, 0, 0)

        self.patchwise_skip_blanks = self.opt.patchwise_skip_blanks
        self.prepare_batches = False
        self.analysis = 'patchwise-grid'

        # set up training/validation
        shuffle_batches = opt.shuffle_batches
        if self.opt.phase == 'train':
            training = True
            validation = False
            self.sample_list_tr = opt.sample_list
            self.set_up(self.sample_list_tr, training, validation, shuffle_batches)
            self.len_dataset = len(self.coord_queue)
            # self.coord_queue.clear()
            self.samples.clear()
        elif self.opt.phase == 'test':
            training = False
            validation = False
            self.sample_list_ts = opt.sample_list
            self.set_up([self.sample_list_ts[opt.eval_sample]], training, validation, shuffle_batches)
        elif self.opt.phase == 'validation':
            training = False
            validation = True
            self.sample_list_ts = opt.sample_list
            self.set_up([self.sample_list_ts[opt.eval_sample]], training, validation, shuffle_batches)
        else:
            print('Invalid phase!')

    # Preprocess data and prepare the batches for a given list of indices
    def set_up(self, indices_list, training=True, validation=False, shuffle=False):
        self.coord_queue = []
        self.samples = {}
        self.cache = {}
        if self.bounding_box: self.samples_bounding_box = {}

        # For normalization
        lr_mean = []
        lr_std = []
        gt_mean = []
        gt_std = []

        # Iterate over all samples
        print('---------------Load data and preprocessing-----------------')
        for i, index in enumerate(indices_list):
            print('[loading case]:', index)
            # Load sample
            if not self.prepare_batches:
                if not training and not validation:
                    # test: load without gt
                    sample = self.opt.data_io.sample_loader(index, load_seg=True, load_pred=False, backup=False, load_gt=False)
                else:
                    sample = self.opt.data_io.sample_loader(index, load_seg=True, load_pred=False, backup=False, load_gt=True)
            # Load sample from backup
            else:
                sample = self.opt.data_io.sample_loader(index, backup=True)

            # find bounding boxes of brain
            if self.bounding_box:
                print('[starting bounding box]...')
                print('training, validation', training, validation)
                if not training:
                    self.cache["shape_before_bounding_" + str(index)] = sample.img_data.shape
                    print("shape_before_bounding_" + str(index), self.cache["shape_before_bounding_" + str(index)])
                x_l1, x_l2, y_l1, y_l2, z_l1, z_l2 = find_bounding_box(sample.img_data)
                # crop the bounding box before normalization
                bb = [x_l1, x_l2, y_l1, y_l2, z_l1, z_l2]
                sample.img_data = sample.img_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                sample.seg_data = sample.seg_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                if not self.opt.phase == 'test':
                    sample.gt_data = sample.gt_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
                self.samples_bounding_box[index] = [x_l1, x_l2, y_l1, y_l2, z_l1, z_l2]
                print(sample.img_data.shape, sample.seg_data.shape, sample.gt_data.shape)

            # Patch slicing before norm, return coordinates of patches
            # IF self.analysis == "patchwise-crop" and training
            print('[starting patch slicing]...')
            if not training:
                self.cache["shape_" + str(index)] = sample.img_data.shape
            coords_img_data = self.analysis_patchwise_grid(sample, training, index)
            self.coord_queue.extend(coords_img_data)
            print('crop patches from: ', i, index, sample.img_data.shape, len(coords_img_data), len(self.coord_queue))

            # Normalization
            print('[starting normalization]...')
            # sf_zscore = Normalization_HCP_SR_FOD(mode="z-score_v2")
            sf_zscore = Normalization_HCP_SR_FOD(mode=self.opt.normalization_mode)
            # Assemble Subfunction classes into a list
            subfunctions = [sf_zscore]
            for sf in subfunctions:
                sf.preprocessing(sample, training=training, validation=validation, opt=self.opt)
                if str(sf.mode) == 'z-score' or 'z-score_v2' or 'z-score_v3' or 'z-score_v4':
                    gt_mean.append(sf.fodgt_mean)
                    gt_std.append(sf.fodgt_std)
                    lr_mean.append(sf.fodlr_mean)
                    lr_std.append(sf.fodlr_std)
            self.sf_zscore = sf_zscore
            self.subfunctions = subfunctions

            # Save the mean and std of all cases during training
            if training:
                np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, sf.mode + '_gt_mean.npy'), np.mean(np.array(gt_mean), axis=0), allow_pickle=True)
                np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, sf.mode + '_gt_std.npy'), np.mean(np.array(gt_std), axis=0), allow_pickle=True)
                np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, sf.mode + '_lr_mean.npy'), np.mean(np.array(lr_mean), axis=0), allow_pickle=True)
                np.save(os.path.join(self.opt.checkpoints_dir, self.opt.name, sf.mode + '_lr_std.npy'), np.mean(np.array(lr_std),axis=0), allow_pickle=True)
            self.samples[index] = sample
        print('-----------end preprocessing-----------------')

    #---------------------------------------------#
    #           Patch-wise grid Analysis          #
    #---------------------------------------------#
    def analysis_patchwise_grid(self, sample, training, index=None):
        # Slice image into patches
        img_to_sliced = sample.img_data
        patches_img, coords_img = slice_matrix(img_to_sliced, self.patch_shape, self.patchwise_overlap,
                                   self.opt.data_io.interface.three_dim, index, save_coords=True)
        # Skip blank patches (only background)
        if training and self.patchwise_skip_blanks:
            # Iterate over each patch
            for i in reversed(range(0, len(patches_img))):
                # IF patch DON'T contain any non background class -> remove it
                if np.sum(patches_img[i]) == 0:
                    del patches_img[i]
                    del coords_img[i]
        # Concatenate a list of patches coordinates into a single numpy array
        coords_img_data = np.stack(coords_img, axis=0)
        # Return preprocessed data tuple
        return coords_img_data


    #---------------------------------------------#
    #          Prediction Postprocessing          #
    #---------------------------------------------#
    # Postprocess prediction data
    def postprocessing(self, sample, prediction, shape=None, coords=None):
        # Reassemble patches into original shape for patchwise analysis
        if self.analysis == "patchwise-crop" or \
            self.analysis == "patchwise-grid":
            # Check if patch was padded
            slice_key = "slicer_" + str(sample)
            if slice_key in self.cache:
                prediction = crop_patch(prediction, self.cache[slice_key])
            # Load cached shape & Concatenate patches into original shape
            # seg_shape = self.cache.pop("shape_" + str(sample))
            seg_shape = self.cache["shape_" + str(sample)]
            prediction = concat_matrices(patches=prediction,
                                    image_size=seg_shape,
                                    window=self.patch_shape,
                                    overlap=self.patchwise_overlap,
                                    three_dim=self.opt.data_io.interface.three_dim,
                                    coords=coords)
        # For fullimages remove the batch axis
        else : prediction = np.squeeze(prediction, axis=0)
        # Run Subfunction postprocessing on the prediction
        for sf in reversed(self.subfunctions):
            prediction = sf.postprocessing(prediction)
        # prediction = self.recover_to_origin_size(prediction, shape)
        if self.bounding_box:
            print('shape_before_bounding_', shape)
            tmp = np.zeros(self.cache.pop("shape_before_bounding_" + str(sample)))
            bb = self.samples_bounding_box[sample]
            tmp[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] = prediction
            prediction = tmp
        # Return postprocessed prediction
        return prediction


    # Return the next batch for associated index
    def __getitem__(self, idx):

        self.coords_batches = self.coord_queue

        if self.opt.phase == 'train':
            # self.samples_list_tr = [self.coords_batches[idx]['index']]
            training = True
            validation = False
            shuffle_batches = self.opt.shuffle_batches
            self.set_up([self.coords_batches[idx]['index']], training, validation, shuffle_batches)
        self.now_sample = self.samples[self.coords_batches[idx]['index']]

        x_start = self.coords_batches[idx]['x_start']
        x_end = self.coords_batches[idx]['x_end']
        y_start = self.coords_batches[idx]['y_start']
        y_end = self.coords_batches[idx]['y_end']
        z_start = self.coords_batches[idx]['z_start']
        z_end = self.coords_batches[idx]['z_end']
        
        self.coords_batches[idx]['dataloader_idx'] = idx

        x = torch.from_numpy(self.now_sample.img_data[x_start:x_end, y_start:y_end, z_start:z_end])
        y = torch.from_numpy(self.now_sample.seg_data[x_start:x_end, y_start:y_end, z_start:z_end])

        # w h d c
        brain = x.clone()
        mask = y.clone()

        # w h d c -> c w h d
        brain = brain.permute(3,0,1,2)
        mask = mask.permute(3,0,1,2)


        if brain.shape[1:-1] != mask.shape[1:-1]:
            print('not the same shape', brain.shape, mask.shape)

        ret = {}
        ret['brain'] = brain
        ret['mask'] = mask

        # IF training / validation, load gt
        if not self.opt.phase == 'test':
            gt = torch.from_numpy(self.now_sample.gt_data[x_start:x_end, y_start:y_end, z_start:z_end])
            gt = gt.permute(3,0,1,2)
            ret['gt'] = gt

        return ret


    # Return the number of batches for one epoch
    def __len__(self):
        # return len(self.coord_queue)
        return self.len_dataset

    # At every epoch end: Shuffle batchPointer list and reset sample_list
    def on_epoch_end(self):
        if self.shuffle and self.training:
            if self.preprocessor.prepare_batches:
                np.random.shuffle(self.batchpointers)
            else:
                np.random.shuffle(self.sample_list)

    def modify_commandline_options(parser, is_train):
        """
        Add any new dataset-specific options, and rewrite default values for existing options.
        """
        parser.add_argument('--test_fold', default='1', type=int, help='test fold number')
        parser.add_argument('--shuffle_batches', default=True, type=bool, help='whether batch order should be shuffled or not ?')
        parser.add_argument('--iterations', default=None, type=int, help='Number of iterations (batches) in a single epoch.')

        parser.add_argument('--patchwise_skip_blanks', default=False, type=bool, help='skip blank patches or not ?')
        parser.add_argument('--gtroot', default=None, type=str, help='path to gt.')
        parser.add_argument('--bounding_box', default=False, type=str, help='use bounding box or not ?')
        parser.add_argument('--normalization_mode', default='z-score_v2', type=str, help='use bounding box or not ?')

        return parser
