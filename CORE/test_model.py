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
from utils.metrics import *
from utils.logger import setup_logger

from data_loading.interfaces import FOD_RE_interface
from data_loading import FOD_RE_Data_IO

import torch
import subprocess
import time
from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
import torch.nn as nn
import numpy as np
import random

# Setup logger
logger = setup_logger()

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# device = torch.device('cpu')

# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.cuda.manual_seed_all(1)
# torch.cuda.manual_seed(1)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

def log_and_record(msg):
    logger.info(msg)
    return msg + '\n'

def main():
    opt = TestOptions().parse()  # get test options

    # Hard-coded test settings
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.patchwise_skip_blanks = False
    opt.bounding_box = False

    # Load sample list
    if (opt.foldroot is not None):
        # Obtain training and validation data set for all folds
        training, validation = load_csv2fold(os.path.join(opt.foldroot, "fold_" + str(opt.test_fold), "sample_list.csv"))
        training.sort()
        sample_list = validation
    elif opt.indexroot is not None:
        training, validation = load_csv2fold(opt.indexroot)
        training.sort()
        sample_list = validation
    else:
        raise ValueError("Please provide either --foldroot or --indexroot")

    logger.info(f"Total samples to evaluate: {len(sample_list)}")

    # Initialize Data IO Interface for NIfTI data
    interface = FOD_RE_interface(channels=opt.input_nc, classes=opt.output_nc,
                                        pattern=opt.index_pattern, suffix=opt.sample_suffix,
                                        gt_suffix=opt.sample_gt_suffix)
    # Create Data IO object to load and write samples in the file structure
    data_io = FOD_RE_Data_IO(interface, input_path=opt.dataroot, output_path=opt.output_dir,
                                batch_path="fod_norm_tensor", delete_batchDir=False, mask_path=opt.maskroot,
                                gt_path=opt.gtroot, sample_list=sample_list)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    sample_list.sort()

    torch.cuda.empty_cache()

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.eval()

    # Create normalization
    # sf_zscore = Normalization_HCP_SR_FOD(mode=opt.normalization_mode)
    # sf = [sf_zscore]
    
    # Obtain training and validation data set
    # opt.sample_list = sample_list
    opt.sample_list = [sample_list[0]]
    # opt.eval_sample = 0
    opt.data_io = data_io
    # save_prediction = True

    loss_list, mse_lst, mae_lst, psnr, time_spent = [], [], [], [], []

    # Create a summary file
    os.makedirs(data_io.output_path, exist_ok=True)
    summary_path = os.path.join(data_io.output_path, f"fold_{opt.test_fold}_summary.txt")
    summary_file = open(summary_path, 'w')

    entry = 'Load checkpoint from: ' + opt.checkpoints_dir + opt.name + '\n'


    for i, index in enumerate(opt.sample_list):
        start_time = time.time()

        # index = sample_list[i]
        # entry += f'case: {index}\n'
        # logger.info(f'[Inference] Case {i+1}/{len(sample_list)}: {index}')
        entry += log_and_record(f"[Inference] Case {i+1}/{len(sample_list)}: {index}")

        # Create dataloader: do normalize + slice patches
        opt.eval_sample = i
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

        # Get normalized images / masks / samples
        brain_img = dataset.dataset.sf_zscore.brain_img
        brain_mask = dataset.dataset.sf_zscore.brain_mask
        ref_img = dataset.dataset.sf_zscore.ref_img
        sample = dataset.dataset.sf_zscore.sample
        mean, std = dataset.dataset.sf_zscore.mean, dataset.dataset.sf_zscore.std

        pred_list, val_loss = [], 0.0

        for j, data in enumerate(dataset):
            model.set_input(data)
            output = model.test()  # run inference

            losses = model.get_current_losses()
            val_loss += losses['R']

            output = output.detach().cpu().numpy()
            # b c w h d => b w h d c
            output = np.transpose(output, (0, 2, 3, 4, 1))
            pred_list.append(output)

        # Postprocess predicted patches
        pred_seg = np.concatenate(pred_list, axis=0)
        pred_seg = dataset.dataset.postprocessing(index, pred_seg, coords=dataset.dataset.coord_queue)

        # Compute testing loss
        avg_loss = val_loss / (j + 1)
        loss_list.append(avg_loss)
        entry += log_and_record(f"[Loss] Validation Loss: {avg_loss:.6f}")
        # logger.info(f"[Loss] Validation Loss: {avg_loss:.6f}")       
        # print(f'[Loss] Validation Loss: {val_loss / (j + 1):.6f}')

        # Compute result image
        result_img = pred_seg.squeeze()
        # Scaling all voxels back to FOD range
        result_img = result_img * std + mean
        ref_img = ref_img * std + mean
        # Set background to 0
        result_img = brain_mask * (1 - brain_mask) + result_img * brain_mask
        ref_img = brain_mask * (1 - brain_mask) + ref_img * brain_mask

        ref_patch, result_patch = ref_img * brain_mask, result_img * brain_mask
        valid_no = np.sum(brain_mask)  # valid voxels

        # Compute metrics
        org_mae = np.sum(np.abs(ref_patch - result_patch)) / valid_no
        mae_lst.append(org_mae)

        sub_mae, sub_mse, sub_psnr = psnr2(ref_patch, result_patch, valid_no)
        mse_lst.append(sub_mse)
        psnr.append(sub_psnr)
        time_spent.append(time.time() - start_time)

        # entry += ('MSE: ' + str(sub_mse)[:6] + ' PSNR: ' + str(sub_psnr)[:6] + '\n')
        entry += log_and_record(f"[Metrics] MSE: {sub_mse:.6f}, PSNR: {sub_psnr:.2f} dB, MAE: {org_mae:.6f}")
        # entry += f"[Metrics] MSE: {sub_mse:.6f}, PSNR: {sub_psnr:.2f} dB, MAE: {org_mae:.6f}\n"
        # logger.info(f"[Metrics] MSE: {sub_mse:.6f}, PSNR: {sub_psnr:.2f} dB, MAE: {org_mae:.6f}")

        # Backup predictions
        dataset.dataset.opt.data_io.save_prediction(result_img, index, info=sample.info)

        entry += log_and_record(f"[{i+1}/{len(sample_list)}] ✓ Inference complete for case: {index}")
        # logger.info(f"[{i+1}/{len(sample_list)}] ✓ Inference complete for case: {index}")

    mean_mse_info = 'Mean MSE: {:.6f} ± {:.6f}'.format(np.mean(mse_lst), np.std(mse_lst))
    mean_mae_info = 'Mean MAE: {:.6f} ± {:.6f}'.format(np.mean(mae_lst), np.std(mae_lst))
    mean_psnr_info = 'Mean PSNR: {:.4f} ± {:.4f}'.format(np.mean(psnr), np.std(psnr))
    mean_time = 'Average inference time per case: {:.4f} seconds'.format(np.mean(time_spent))

    entry += log_and_record("========== Summary ==========")
    entry += log_and_record(mean_mse_info)
    entry += log_and_record(mean_mae_info)
    entry += log_and_record(mean_psnr_info)
    entry += log_and_record(mean_time)
    entry += log_and_record('For record keeping: {:.4f}±{:.4f} {:.4f}±{:.4f} {:.4f}±{:.4f}'.format(np.mean(mse_lst), np.std(mse_lst), 
                                                                                          np.mean(mae_lst), np.std(mae_lst),
                                                                                          np.mean(psnr), np.std(psnr)))

    summary_file.write(entry)
    summary_file.close()

if __name__ == '__main__':
    main()