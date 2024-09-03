"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from utils.utils import *

from data_loading.interfaces import HCP_SR_v2_NIFTI_interface
from data_loading import HCP_SR_v2_Data_IO
from processing.subfunctions import Normalization_HCP_SR_FOD

import torch
import subprocess
import time
from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn as nn
import numpy as np
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# device = torch.device('cpu')

# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.cuda.manual_seed_all(1)
# torch.cuda.manual_seed(1)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    # opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.patchwise_skip_blanks = False
    opt.bounding_box = False
    istest = True
    # load sample list
    if (opt.foldroot is not None) and istest==False:
        # Obtain training and validation data set for all folds
        training, validation = load_csv2fold(os.path.join(opt.foldroot, "fold_" + str(opt.test_fold), "sample_list.csv"))
        training.sort()
        # sample_list = training
        sample_list = validation
    elif opt.indexroot is not None:
        training, validation = load_csv2fold(opt.indexroot)
        training.sort()
        # sample_list = training
        sample_list = validation
    elif (opt.foldroot is not None) and istest==True:
        # load all data of this dataset
        training, validation = load_csv2fold(os.path.join(opt.foldroot, "fold_" + str(opt.test_fold), "sample_list.csv"))
        training.sort()
        # sample_list = training
        sample_list = training + validation
    else:
        # ---- no sample list
        sample_list = None # Intailize with Data_IO class

    print(sample_list, len(sample_list))

    # Initialize Data IO Interface for NIfTI data
    # interface = HCP_SR_v2_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc,
    #                                       pattern='^\d{3}$', suffix="_wmfod_norm.nii.gz")
    # IF load mif of MND
    # interface = HCP_SR_v2_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc,
    #                                       pattern='^\d{3}$', suffix="_wmfod_norm.mif.gz")
    interface = HCP_SR_v2_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc,
                                          pattern=opt.index_pattern, suffix=opt.sample_suffix,
                                          gt_suffix=opt.sample_gt_suffix)
    # Create Data IO object to load and write samples in the file structure
    data_io = HCP_SR_v2_Data_IO(interface, input_path=opt.dataroot, output_path=opt.output_dir,
                                batch_path="fod_norm_tensor", delete_batchDir=False, mask_path=opt.maskroot,
                                gt_path=opt.gtroot, sample_list=sample_list)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    sample_list.sort()
    # sample_list = ['077']

    # Create a summary file
    if not os.path.exists(data_io.output_path):
        subprocess.call(['mkdir', data_io.output_path])
    if os.path.exists(os.path.join(data_io.output_path, "fold_" + str(opt.test_fold) + '_summary.txt')):
        subprocess.call(['rm', os.path.join(data_io.output_path, 'summary.txt')])
    summary_file = open(os.path.join(data_io.output_path, "fold_" + str(opt.test_fold) + '_summary.txt'), 'w')

    entry = 'Load checkpoint from: ' + opt.checkpoints_dir + opt.name + '\n'

    torch.cuda.empty_cache()

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()

    # Create normalization
    sf_zscore = Normalization_HCP_SR_FOD(mode=opt.normalization_mode)
    sf = [sf_zscore]

    # Obtain training and validation data set
    # opt.sample_list = [sample_list[0]]
    opt.sample_list = sample_list
    # opt.sample_list = ['077']
    opt.eval_sample = 0
    opt.data_io = data_io
    save_prediction = True

    loss_list = []
    mse_lst = []
    mae_lst = []
    psnr = []
    mse_med = []
    psnr_med = []
    time_spent = []

    # gt_mean = np.load(os.path.join(opt.checkpoints_dir, opt.name, sf_zscore.mode + '_gt_mean.npy'), allow_pickle=True)
    # gt_std = np.load(os.path.join(opt.checkpoints_dir, opt.name, sf_zscore.mode + '_gt_std.npy'), allow_pickle=True)
    # lr_mean = np.load(os.path.join(opt.checkpoints_dir, opt.name, sf_zscore.mode + '_lr_mean.npy'), allow_pickle=True)
    # lr_std = np.load(os.path.join(opt.checkpoints_dir, opt.name, sf_zscore.mode + '_lr_std.npy'), allow_pickle=True)
    # mean = gt_mean
    # std = gt_std
    
    for i in range(len(opt.sample_list)):
        start_time = time.time()

        index = sample_list[i]
        entry += ('case: ' + index + '\n')
        print('case:', i, index)

        # Create dataloader: do normalize + slice patches
        opt.eval_sample = i
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

        # Get normalized images / masks / samples
        brain_img = dataset.dataset.sf_zscore.brain_img
        brain_mask = dataset.dataset.sf_zscore.brain_mask
        ref_img = dataset.dataset.sf_zscore.ref_img
        sample = dataset.dataset.sf_zscore.sample
        mean = dataset.dataset.sf_zscore.mean
        std = dataset.dataset.sf_zscore.std

        pred_list = []
        val_loss = 0.
        for j, data in enumerate(dataset):
            model.set_input(data)
            output = model.test()  # run inference
            losses = model.get_current_losses()
            val_loss += losses['R']

            output = output.detach().cpu().numpy()
            # b c w h d -> b w h d c
            output = np.transpose(output, (0, 2, 3, 4, 1))
            temp = output
            pred_list.append(temp)


        # Postprocess predicted patches
        pred_seg = np.concatenate(pred_list, axis=0)
        pred_seg = dataset.dataset.postprocessing(sample_list[opt.eval_sample], pred_seg, coords=dataset.dataset.coords_batches)

        # Compute testing loss
        loss_list.append(val_loss / (j + 1))
        print('[Test] loss:', val_loss / (j + 1))

        # Compute result image
        result_img = pred_seg.squeeze()

        # Scaling all voxels back to FOD range
        result_img = result_img * std + mean
        # brain_mask = np.expand_dims(brain_mask, 3).repeat(45, axis=3)
        ref_img = ref_img * std + mean
        # print('std.shape, mean.shape', std.shape, mean.shape)
        # Set background to 0
        result_img = brain_mask * (1 - brain_mask) + result_img * brain_mask
        ref_img = brain_mask * (1 - brain_mask) + ref_img * brain_mask

        ref_patch = ref_img * brain_mask
        result_patch = result_img * brain_mask
        valid_no = np.sum(brain_mask)  # valid pixels

        # compute metrics
        org_mae = np.sum(np.abs(ref_patch - result_patch)) / valid_no
        mae_lst.append(org_mae)
        sub_mae, sub_mse, sub_psnr = psnr2(ref_patch, result_patch, valid_no)
        entry += ('MSE: ' + str(sub_mse)[:6] + ' PSNR: ' + str(sub_psnr)[:6] + '\n')
        mse_lst.append(sub_mse)
        psnr.append(sub_psnr)

        time_spent.append(time.time() - start_time)
        print('MSE: ', sub_mse, 'PSNR: ', sub_psnr)

        # Backup predicted segmentation
        if save_prediction:
            dataset.dataset.opt.data_io.save_prediction(result_img, sample_list[i], info=sample.info)
        print('***************************End inference******************************')

    mean_mse_info = 'mean mse: {:.6f}±{:.6f}'.format(np.mean(mse_lst), np.std(mse_lst))
    mean_mae_info = 'mean mae: {:.6f}±{:.6f}'.format(np.mean(mae_lst), np.std(mae_lst))
    mean_psnr_info = 'mean psnr: {:.4f}±{:.4f}'.format(np.mean(psnr), np.std(psnr))
    mean_time = 'average case time: {:.4f} seconds'.format(np.mean(time_spent))
    print(mean_mse_info)
    print(mean_mae_info)
    print(mean_psnr_info)
    print(mean_time)

    print('For record keeping: \n {:.4f}±{:.4f} {:.4f}±{:.4f} {:.4f}±{:.4f}' \
          .format(np.mean(mse_lst), np.std(mse_lst), np.mean(mae_lst), np.std(mae_lst), \
                  np.mean(psnr), np.std(psnr)))

    entry += (mean_mse_info + '\n')
    entry += (mean_psnr_info + '\n')
    entry += '\n'

    summary_file.write(entry)
    summary_file.close()
