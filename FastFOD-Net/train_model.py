"""
Training script for LG-Net.

See more details at https://github.com/jackjacktang/LG-Net/
"""
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from utils.utils import *
import torch
import os
import numpy as np

from data_loading.interfaces import HCP_SR_v2_NIFTI_interface
from data_loading import HCP_SR_v2_Data_IO
from processing.subfunctions import Normalization_HCP_SR_FOD

import matplotlib.pyplot as plt
import random
import subprocess

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# device = torch.device('cpu')

# torch.manual_seed(1)
# np.random.seed(1)
# random.seed(1)
# torch.cuda.manual_seed_all(1)
# torch.cuda.manual_seed(1)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# torch.manual_seed(4)
# np.random.seed(4)
# random.seed(4)
# torch.cuda.manual_seed_all(4)
# torch.cuda.manual_seed(4)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# torch.manual_seed(2)
# np.random.seed(2)
# random.seed(2)
# torch.cuda.manual_seed_all(2)
# torch.cuda.manual_seed(2)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# torch.manual_seed(3)
# np.random.seed(3)
# random.seed(3)
# torch.cuda.manual_seed_all(3)
# torch.cuda.manual_seed(3)
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True

# Create an evaluation subdirectory and change path
def create_directories( eval_path, subeval_path=None):
    # Create evaluation directory if necessary
    if not os.path.exists(eval_path):
        os.mkdir(eval_path)
    # Create evaluation subdirectory if necessary
    if subeval_path is not None:
        # Concatenate evaluation subdirectory path if present
        subdir = os.path.join(eval_path, subeval_path)
        # Set up the evaluation subdirectory
        if not os.path.exists(subdir):
            os.mkdir(subdir)
        # Return path to evaluation subdirectory
        return subdir

# -----------------------------------------------------#
#                   CSV Management                    #
# -----------------------------------------------------#
# Subfunction for writing a fold sampling to disk
def write_fold2csv(file_path, training, validation):
    with open(file_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter=" ")
        writer.writerow(["TRAINING:"] + list(training))
        writer.writerow(["VALIDATION:"] + list(validation))

# Subfunction for loading a fold sampling from disk
def load_csv2fold(file_path):
    training = None
    validation = None
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=" ")
        for row in reader:
            if not training:
                training = row[1:]
            else:
                validation = row[1:]
    return training, validation


def split_folds(sample_list, k_fold=5, evaluation_path="evaluation"):
    # Randomly permute the sample list
    # samples_permuted = np.random.permutation(sample_list)
    samples_permuted = sample_list
    # Split sample list into folds
    folds = np.array_split(samples_permuted, k_fold)
    fold_indices = list(range(len(folds)))
    # Iterate over each fold
    for i in fold_indices:
        # Subset training and validation data set
        training = np.concatenate([folds[x] for x in fold_indices if x != i],
                                  axis=0)
        validation = folds[i]
        # Initialize evaluation subdirectory for current fold
        subdir = create_directories(evaluation_path, "fold_" + str(i))
        fold_cache = os.path.join(subdir, "sample_list.csv")
        # Write sampling to disk
        write_fold2csv(fold_cache, training, validation)

        print('training, validation:', training, validation, training.shape, validation.shape)


torch.manual_seed(5)
np.random.seed(5)
random.seed(5)
torch.cuda.manual_seed_all(5)
torch.cuda.manual_seed(5)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    opt = TrainOptions().parse()   # get training options

    # load sample list
    if opt.foldroot is not None:
        # Obtain training and validation data set for all folds
        training, validation = load_csv2fold(os.path.join(opt.foldroot, "fold_" + str(opt.test_fold), "sample_list.csv"))
        training.sort()
        sample_list = training
    elif opt.indexroot is not None:
        # validation = ['147737', '211720', '672756', '161731', '130013', '144832', '133019', '103818', '208226', '160123',
        #               '199655','298051', '159340', '239944', '149741', '113922', '103414', '129028', '105014', '127933']
        # training = ['140925', '138534', '196750', '756055', '118528', '113619', '101107', '127630', '899885', '131217',
        #         '139637', '111312', '176542', '100307', '188347', '135932', '122620', '201111', '154734', '151627',
        #         '751348', '123925', '149539', '857263', '103111', '366446', '130316', '148840', '654754', '163129',
        #         '128632', '123117', '211417', '190031', '397760', '212318', '136833', '148335', '101309', '128127',
        #         '221319', '125525', '111716', '156637', '146432', '414229', '856766', '245333', '153025', '135225',
        #         '122317', '151526', '151223', '149337', '124422', '116524', '115320', '214423', '117122', '105115',
        #         '280739', '101915', '100408', '198451', '178950', '120111', '189450', '114419', '162733', '118932',
        #         '499566', '110411', '192540', '133928', '126325', '106016', '108828', '792564', '118730', '131722']
        # write_fold2csv(opt.indexroot, training, validation)
        training, validation = load_csv2fold(opt.indexroot)
        # training.sort()
        sample_list = training
    else:
        sample_list = None # Intailize with Data_IO class

        # with open("/home/xinyi/MSBIR/sub_list/multi_shell_subs_bl.txt", "r") as file:
        #     # Read the lines and store them in a list
        #     sub_ids = file.readlines()
        #
        # # Strip newline characters from each line
        # sub_list = [sub_id.strip().replace("/", "_") for sub_id in sub_ids]
        # # Print the list of sub IDs
        #
        src_path = '/home/xinyi/MSBIR/fod/'
        name = 'fod_bbox'
        sub_list = []
        for s in os.listdir(src_path):
            if not s.endswith("_WMfod_norm.mif.gz"): continue
            sub_list.append(s[:-len("_WMfod_norm.mif.gz")])
            # break
        print(sub_list, len(sub_list))
        sample_list = sorted(sub_list)
        print(sample_list, len(sample_list))


    # Initialize Data IO Interface for NIfTI data
    # We are using 4 classes due to [background, brain, lesion]
    # IF load nifti of MND
    # interface = HCP_SR_v2_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc,
    #                                       pattern='^\d{3}$', suffix="_wmfod_norm.nii.gz")
    # IF load mif of MND
    # interface = HCP_SR_v2_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc,
    #                                       pattern='^\d{3}$', suffix="_wmfod_norm.mif.gz")
    interface = HCP_SR_v2_NIFTI_interface(channels=opt.input_nc, classes=opt.output_nc,
                                          pattern=opt.index_pattern, suffix=opt.sample_suffix,
                                          gt_suffix=opt.sample_gt_suffix)
    # Create Data IO object to load and write samples in the file structure
    data_io = HCP_SR_v2_Data_IO(interface, input_path=opt.dataroot, output_path="predictions_"+str(opt.name),
                                batch_path="fod_norm_tensor", delete_batchDir=False, mask_path=opt.maskroot,
                                gt_path=opt.gtroot, sample_list=sample_list)

    # Access all available samples in our file structure
    sample_list = data_io.get_indiceslist()
    # sample_list.sort()
    random.shuffle(sample_list)
    print('sample_list:', sample_list, len(sample_list))

    if opt.phase == 'train':
        torch.cuda.empty_cache()

        opt.data_io = data_io
        opt.patchwise_skip_blanks = True
        opt.sample_list = sample_list
        opt.sample_list = [sample_list[0]]
        # opt.sample_list = [sample_list[0], sample_list[1]]
        # print('sample_list for train:', opt.sample_list)
        # opt.serial_batches = True

        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)    # get the number of images in the dataset.
        print('The number of training images = %d' % dataset_size)

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

            # training
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
                    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)

                iter_data_time = time.time()
                # only load half number of patches
                # if i >= len(dataset)/(opt.batch_size*2):
                #     print('randop at i', i)
                #     break

            model.update_learning_rate()  # update learning rates at the end of every epoch.
            loss_list.append(tot_loss / (i+1))
            print('[TRAIN] loss:', tot_loss / (i+1))

            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    elif opt.phase == 'splitfolds':
        split_folds(sample_list, k_fold=5, evaluation_path="/home/xinyi/MSBIR/folds")
    else:
        print('Invalid phase!')