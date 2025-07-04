"""
Training script.
"""

import os
import csv
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.utils import *
from data_loading.interfaces import FOD_RE_interface
from data_loading import FOD_RE_Data_IO
from utils.logger import setup_logger

# Setup logger
logger = setup_logger()

# Allow duplicated OpenMP library loading
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# device = torch.device('cpu')

# -----------------------------------------------------#
#                Seed for Reproducibility              #
# -----------------------------------------------------#
torch.manual_seed(5)
np.random.seed(5)
random.seed(5)
torch.cuda.manual_seed_all(5)
torch.cuda.manual_seed(5)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# -----------------------------------------------------#
#                    Main Script                       #
# -----------------------------------------------------#
if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options

    # Initialize Data IO Interface for FOD data
    interface = FOD_RE_interface(
        channels=opt.input_nc,
        classes=opt.output_nc,
        pattern=opt.index_pattern,
        suffix=opt.sample_suffix,
        gt_suffix=opt.sample_gt_suffix
    )    

    if opt.phase == 'train':
        # Load sample list
        if opt.foldroot is not None:
            # Obtain training and validation data set for all folds
            training, validation = load_csv2fold(os.path.join(opt.foldroot, "fold_" + str(opt.test_fold), "sample_list.csv"))
            training.sort()
            sample_list = training
            logger.info(f"Loaded training samples from fold_{opt.test_fold}, total: {len(sample_list)}")
        elif opt.indexroot is not None:
            training, validation = load_csv2fold(opt.indexroot)
            sample_list = training
            logger.info(f"Loaded training samples from indexroot, total: {len(sample_list)}")
        else:
            sample_list = None # Intailize with Data_IO class
            logger.warning("No sample list provided (foldroot/indexroot is None).") 

        # Create Data IO object to load and write samples in the file structure
        data_io = FOD_RE_Data_IO(
            interface,
            input_path=opt.dataroot,
            output_path=f"predictions_{opt.name}",
            batch_path="fod_norm_tensor",
            delete_batchDir=False,
            mask_path=opt.maskroot,
            gt_path=opt.gtroot,
            sample_list=sample_list
        )

        # Access all available samples in our file structure
        sample_list = data_io.get_indiceslist()
        random.shuffle(sample_list)
        logger.info(f"Final sample list size after shuffle: {len(sample_list)}")
    
        torch.cuda.empty_cache()

        opt.data_io = data_io
        opt.patchwise_skip_blanks = True
        opt.sample_list = sample_list
        opt.sample_list = [sample_list[0]]

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        logger.info(f"The number of training images: {dataset_size}")

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        recorder = Recorder(opt)

        total_iters = 0                # the total number of training iterations
        loss_list = []
        
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            tot_loss = 0.
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            # logger.info(f"Start epoch {epoch}/{opt.n_epochs + opt.n_epochs_decay}")

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                model.set_input(data)
                model.optimize_parameters()
                losses = model.get_current_losses()
                tot_loss += losses['R']

                if total_iters % opt.display_freq == 0:
                    save_result = total_iters == 0
                    recorder.plot_current_losses(total_iters, losses)

                if total_iters % opt.print_freq == 0:
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    t_data = iter_start_time - iter_data_time
                    recorder.print_current_losses(epoch, total_iters, losses, t_comp, t_data)

                if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                    # print(f"[Checkpoint] Saving latest model (epoch {epoch}, iters {total_iters})")
                    logger.info(f"[Checkpoint] Saved latest model at epoch {epoch}, iter {total_iters}")
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

            model.update_learning_rate()  # update learning rates at the end of every epoch.
            avg_loss = tot_loss / (i + 1)
            loss_list.append(avg_loss)
            # print(f"[TRAIN] Epoch {epoch} - Loss: {avg_loss:.6f}")
            logger.info(f"[TRAIN] Epoch {epoch} - Loss: {avg_loss:.6f}")

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                # print(f"[Checkpoint] Saving model at epoch {epoch}, total_iters {total_iters}")
                logger.info(f"[Checkpoint] Saving model at epoch {epoch}, total_iters {total_iters}")
                model.save_networks('latest')
                model.save_networks(epoch)

            logger.info(f'End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} | Time Taken: {int(time.time() - epoch_start_time)} sec')
            # print('End of epoch %d / %d | Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))

    elif opt.phase == 'splitfolds':
        sample_list = interface.initialize(input_path=opt.dataroot)   
        logger.info(f"[splitfolds] Found {len(sample_list)} samples for splitting.")

        if not sample_list:
            logger.error("[splitfolds] No samples found. Please check dataroot or file naming.")
        else:
            split_folds(sample_list, k_fold=5, evaluation_path=opt.foldroot)
            logger.info(f"[splitfolds] Fold split completed at: {opt.foldroot}")

    else:
        logger.error("[Error] Invalid phase specified in options.")