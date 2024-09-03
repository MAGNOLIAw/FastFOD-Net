"""
CSD Super Resolution
Rui Zeng All rights reserved.
Written by Rui Zeng @ USyd Brain and Mind Centre. r.zeng@outlook.com / rui.zeng@sydney.edu.au
Any use of these scripts should be permitted by Rui Zeng.
"""
from __future__ import print_function
from abc import ABC
from numpy.core.defchararray import index
import torch
import numpy as np
from PIL import Image
import os
import random
import skimage
import skimage.transform
from distutils.version import LooseVersion
import torchvision.transforms as transforms
from torch.autograd import Variable
import numbers
import numpy as np
from sklearn.utils import check_array, check_random_state
from numpy.lib.stride_tricks import as_strided
from itertools import product
from typing import Tuple, List
import nibabel as nib
from mrtrix3pyio.mrtrix3.io import load_mrtrix, save_mrtrix
import pickle
import math
from tqdm import trange
import sys
import subprocess
from IPython.display import clear_output


# Deprecated class
class FixelMetric():
    def __init__(self, max_fixel_per_voxel):
        self.fiber_sum = np.zeros(max_fixel_per_voxel)
        self.fiber_hit = np.zeros(max_fixel_per_voxel)
        self.fiber_afd_error = []
        self.fiber_directions_error = []
        self.fiber_disp_error = []
        self.fiber_peak_error = []
        self.fiber_gt_afd = []
        self.fiber_pred_afd = []
        for i in range(max_fixel_per_voxel):
            self.fiber_afd_error.append([])
            self.fiber_directions_error.append([])
            self.fiber_disp_error.append([])
            self.fiber_peak_error.append([])
            self.fiber_gt_afd.append([])
            self.fiber_pred_afd.append([])
    def add1(self, nth_fixel):
        self.fiber_sum[nth_fixel-1] = self.fiber_sum[nth_fixel-1] + 1

    def hit(self, nth_fixel, flag):
        if flag:
            self.fiber_hit[nth_fixel-1] = self.fiber_hit[nth_fixel-1] + 1

    def afd_error(self, pred_afd, gt_afd, nth_fixel, flag):
        if flag:
            self.fiber_afd_error[nth_fixel - 1].append(np.abs(pred_afd - gt_afd))

    def afd_stats(self, pred_afd, gt_afd, nth_fixel, flag):
        if flag:
            self.fiber_gt_afd[nth_fixel - 1].append(np.abs(gt_afd))
            self.fiber_pred_afd[nth_fixel - 1].append(np.abs(pred_afd))

    def peak_error(self, pred_peak, gt_peak, nth_fixel, flag):
        if flag:
            self.fiber_peak_error[nth_fixel - 1].append(np.abs(pred_peak - gt_peak))

    def dispersion_error(self, pred_dispersion, gt_dispersion, nth_fixel, flag):
        if flag:
            self.fiber_disp_error[nth_fixel - 1].append(np.abs(pred_dispersion - gt_dispersion))

    def directions_error(self, pred_directions, gt_directions, nth_fixel, flag):
        if flag:
            self.fiber_directions_error[nth_fixel - 1].append(np.mean(np.abs(angle(pred_directions, gt_directions))))

    def acc(self):
        self.hitacc = self.fiber_hit / (self.fiber_sum)

    def peak(self):
        self.avg_peak = []
        for i, metric in enumerate(self.fiber_peak_error):
            self.avg_peak.append(np.mean(metric))

    def afd(self):
        self.avg_afd = []
        for i, metric in enumerate(self.fiber_afd_error):
            self.avg_afd.append(np.mean(metric))

    def disp(self):
        self.avg_disp = []
        for i, metric in enumerate(self.fiber_disp_error):
            self.avg_disp.append(np.mean(metric))

    def angle(self):
        self.avg_angle =[]
        for i, metric in enumerate(self.fiber_directions_error):
            self.avg_angle.append(np.mean(metric))

    def postprocessing(self):
        self.histogram_angular = []
        for i, metric in enumerate(self.fiber_directions_error):
            self.histogram_angular.append(np.h)


def dotproduct(v1, v2):
  return np.sum(v1 * v2, axis=1)

def length(v):
  return np.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return np.arccos(np.clip(np.abs(dotproduct(v1, v2) / (length(v1) * length(v2))), 0., 1.)) * 180 / np.pi


def align_one(angle_distance):
    shape = angle_distance.shape
    num_fixel_gt = angle_distance.shape[0]
    num_fixel_est = angle_distance.shape[1]
    angle_index = np.zeros(num_fixel_gt, dtype=np.int)
    # if num_fixel_est < num_fixel_gt, a lot of these indices will be 0 by default
    mask = np.zeros((num_fixel_gt, num_fixel_est))
    angle_distance = np.ma.array(angle_distance, mask=mask)
    # for each GT fixel the one with the lowest anguler error
    for i in range(num_fixel_gt):
        flat_index = np.ma.argmin(angle_distance)
        row, col = np.unravel_index(flat_index, shape)
        angle_index[row] = col
        # a vector with the GT fixels that contains the indices that correspond to the best non-gt fixels
        mask[row, :] = 1
        mask[:, col] = 1
        angle_distance = np.ma.array(angle_distance, mask=mask)
    return angle_index

def alignFixelGroup(directions_gt, directions_est, num_fixel_gt, num_fixel_est):
    angle_distance = np.zeros((num_fixel_gt, num_fixel_est))

    for i in range(num_fixel_gt):
        for j in range(num_fixel_est):
            angle_vec = angle(directions_gt[i:i+1, :], directions_est[j:j+1, :])
            angle_distance[i, j] = angle_vec

    return align_one(angle_distance)

def rearrange_matrix(matrix, order):
    matrix_test = matrix[order]
    return matrix_test

def load_fod_sh(fod_filepath):
    """Load the specified fod file and its corresponding spherical harmonics coefficients
    """
    fod_file = nib.load(fod_filepath)
#     fod_sh = fod_file.get_data()
    fod_sh = fod_file.get_fdata()
    fod_affine = fod_file.affine
    fod_header = fod_file.header
    return np.asarray(fod_sh, dtype=np.float32), fod_affine

# angular correlation coefficient
def angular_corr(pred_fod, gt_fod):
    pred_fod = pred_fod[:, :, :, 1:]
    gt_fod = gt_fod[:, :, :, 1:]
    numerator = np.sum(pred_fod * gt_fod, axis=-1)
    denominator = np.sqrt(np.sum(pred_fod**2, axis=-1)) * np.sqrt(np.sum(gt_fod**2, axis=-1))
    return numerator/denominator


def mean_angular_error(index_gt_path, direcions_gt_path, index_est_path, directions_est_path, output_path, max_num_fixel_per_voxel=3):
    index_gt = load_mrtrix(index_gt_path)
    directions_gt = load_mrtrix(direcions_gt_path)

    index_est = load_mrtrix(index_est_path)
    directions_est = load_mrtrix(directions_est_path)

    shape_gt = index_gt.data.shape
    shape_est = index_est.data.shape

    assert shape_gt == shape_est, 'should be the same'

    mae = np.zeros(shape_gt[:3])
    equal_mask = np.zeros(shape_gt[:3])
    est_mask = np.zeros(shape_gt[:3])


    # iterate each voxel in gt_index
    for i in range(shape_gt[0]):
        for j in range(shape_gt[1]):
            for k in range(shape_gt[2]):
                # print('we are at {0}, {1}, {2}'.format(i, j, k))
                num_fixel_gt, fixel_idx_gt = index_gt.data[i, j, k]
                num_fixel_est, fixel_idx_est = index_est.data[i, j, k]

                # gt: ignore regions without fixels
                if num_fixel_gt == 0:
                    continue

                # prediction: ignore regions without fixels
                if num_fixel_est == 0:
                    continue

                if num_fixel_gt == num_fixel_est:
                    equal_mask[i, j, k] = 1

                est_mask[i, j, k] = 1


                fixel_directions_gt = directions_gt.data[fixel_idx_gt : (fixel_idx_gt + num_fixel_gt), :, 0]
                fixel_directions_est = directions_est.data[fixel_idx_est : (fixel_idx_est + num_fixel_est), :, 0]

                aligned_order = alignFixelGroup(fixel_directions_gt, fixel_directions_est, num_fixel_gt, num_fixel_est)
                fixel_directions_est = rearrange_matrix(fixel_directions_est, aligned_order)
                angle_error = angle(fixel_directions_est, fixel_directions_gt)
                idx_low_to_high = angle_error.argsort()
                sorted_angle_error = angle_error[idx_low_to_high]
                sorted_angle_error = sorted_angle_error[:max_num_fixel_per_voxel]
                voxel_mae = np.mean(sorted_angle_error)
                mae[i, j, k] = voxel_mae



    mae_path = os.path.join(output_path, 'mae.mif')
    equal_mask_path = os.path.join(output_path, 'equal_mask.mif')
    est_mask_path = os.path.join(output_path, 'est_mask.mif')

    os.makedirs(output_path, exist_ok=True)

    index_gt.data = mae
    save_mrtrix(mae_path, index_gt)

    index_gt.data = equal_mask
    save_mrtrix(equal_mask_path, index_gt)

    index_gt.data = est_mask
    save_mrtrix(est_mask_path, index_gt)

    return True


def afd_peak(index_gt_path, direcions_gt_path, afd_gt_path, peak_gt_path, index_est_path, directions_est_path, afd_est_path, peak_est_path, output_path, max_num_fixel_per_voxel=3):
    index_gt = load_mrtrix(index_gt_path)
    directions_gt = load_mrtrix(direcions_gt_path)
    afd_gt = load_mrtrix(afd_gt_path)
    peak_gt = load_mrtrix(peak_gt_path)

    index_est = load_mrtrix(index_est_path)
    directions_est = load_mrtrix(directions_est_path)
    afd_est = load_mrtrix(afd_est_path)
    peak_est = load_mrtrix(peak_est_path)

    shape_gt = index_gt.data.shape
    shape_est = index_est.data.shape

    assert shape_gt == shape_est, 'should be the same'

    mae = np.zeros(shape_gt[:3])
    afd_error_volume = np.zeros(shape_gt[:3])
    peak_error_volume = np.zeros(shape_gt[:3])
    equal_mask = np.zeros(shape_gt[:3])
    est_mask = np.zeros(shape_gt[:3])

    # iterate each voxel in gt_index
    for i in range(shape_gt[0]):
        for j in range(shape_gt[1]):
            for k in range(shape_gt[2]):
                # print('we are at {0}, {1}, {2}'.format(i, j, k))
                num_fixel_gt, fixel_idx_gt = index_gt.data[i, j, k]
                num_fixel_est, fixel_idx_est = index_est.data[i, j, k]

                if num_fixel_gt == 0:
                    continue

                if num_fixel_est == 0:
                    continue

                if num_fixel_gt == num_fixel_est:
                    equal_mask[i, j, k] = 1

                est_mask[i, j, k] = 1

                common_num_fixel = np.min([max_num_fixel_per_voxel, num_fixel_gt, int(num_fixel_est)])

                fixel_directions_gt = directions_gt.data[fixel_idx_gt : (fixel_idx_gt + num_fixel_gt), :, 0]
                fixel_directions_est = directions_est.data[fixel_idx_est : (fixel_idx_est + num_fixel_est), :, 0]

                fixel_peak_gt = peak_gt.data[fixel_idx_gt : (fixel_idx_gt + num_fixel_gt), :, 0]
                fixel_peak_est = peak_est.data[fixel_idx_est : (fixel_idx_est + num_fixel_est), :, 0]

                fixel_afd_gt = afd_gt.data[fixel_idx_gt : (fixel_idx_gt + num_fixel_gt), :, 0]
                fixel_afd_est = afd_est.data[fixel_idx_est : (fixel_idx_est + num_fixel_est), :, 0]

                aligned_order = alignFixelGroup(fixel_directions_gt, fixel_directions_est, num_fixel_gt, num_fixel_est)
                fixel_directions_est = rearrange_matrix(fixel_directions_est, aligned_order)

                fixel_peak_est = rearrange_matrix(fixel_peak_est, aligned_order)
                fixel_afd_est = rearrange_matrix(fixel_afd_est, aligned_order)

                angle_error = angle(fixel_directions_est, fixel_directions_gt)
                idx_low_to_high = angle_error.argsort()
                sorted_angle_error = angle_error[idx_low_to_high]
                sorted_angle_error = sorted_angle_error[:common_num_fixel]
                voxel_mae = np.mean(sorted_angle_error)

                peak_error = np.abs(fixel_peak_gt - fixel_peak_est)
                afd_error = np.abs(fixel_afd_gt - fixel_afd_est)
                sorted_peak_error = peak_error[idx_low_to_high]
                sorted_afd_error = afd_error[idx_low_to_high]
                sorted_peak_error = sorted_peak_error[:common_num_fixel]
                sorted_afd_error = sorted_afd_error[:common_num_fixel]

                voxel_peak_error = np.mean(sorted_peak_error)
                voxel_afd_error = np.mean(sorted_afd_error)



                mae[i, j, k] = voxel_mae
                afd_error_volume[i, j, k] = voxel_afd_error
                peak_error_volume[i, j, k] = voxel_peak_error



    mae_path = os.path.join(output_path, 'mae.mif')
    peak_path = os.path.join(output_path, 'peak_error.mif')
    afd_path = os.path.join(output_path, 'afd_error.mif')
    equal_mask_path = os.path.join(output_path, 'equal_mask.mif')
    est_mask_path = os.path.join(output_path, 'est_mask.mif')

    os.makedirs(output_path, exist_ok=True)

    index_gt.data = mae
    save_mrtrix(mae_path, index_gt)

    index_gt.data = afd_error_volume
    save_mrtrix(afd_path, index_gt)

    index_gt.data = peak_error_volume
    save_mrtrix(peak_path, index_gt)

    index_gt.data = equal_mask
    save_mrtrix(equal_mask_path, index_gt)

    index_gt.data = est_mask
    save_mrtrix(est_mask_path, index_gt)

    return True

def get_sub_list(test_path):
    # test_path = '/home/data/HCP_SR/HCP_50/test/'
    sub_list=[]
    for s in os.listdir(test_path):
        sub_list.append(s)
    print(sub_list)
    return sub_list



def nii_to_fixel(sub_list, nii_path, mif_path, fixel_path, ms=True):
    if not os.path.exists(mif_path):
        os.mkdir(mif_path)
    if not os.path.exists(fixel_path):
        os.mkdir(fixel_path)

    sub_ctr=1
    for sub in sub_list:
        print('process: {}, progress:{}/{}'.format(sub, sub_ctr, len(sub_list)))

        if ms:
            # nii = os.path.join(nii_path, sub, 'msmt_csd', 'SR_WM_FODs_normalised.nii.gz')
            # mif_file = os.path.join(mif_path, sub, 'msmt_csd', 'SR_WM_FODs_normalised.mif.gz')
            nii = os.path.join(nii_path, sub, 'msmt_csd', 'WM_FODs_normalised.nii.gz')
            mif_file = os.path.join(mif_path, sub, 'msmt_csd', 'WM_FODs_normalised.mif.gz')
            if not os.path.exists(os.path.join(mif_path, sub, 'msmt_csd')):
                os.mkdir(os.path.join(mif_path, sub))
                os.mkdir(os.path.join(mif_path, sub, 'msmt_csd'))
        else:
            nii = os.path.join(nii_path, sub, 'ss3t_csd', 'WM_FODs_normalised.nii.gz')
            mif_file = os.path.join(mif_path, sub, 'ss3t_csd', 'WM_FODs_normalised.mif.gz')
            if not os.path.exists(os.path.join(mif_path, sub, 'ss3t_csd')):
#                 os.mkdir(os.path.join(mif_path, sub))
                os.mkdir(os.path.join(mif_path, sub, 'ss3t_csd'))

        subprocess.call(['mrconvert', nii, mif_file])
        if ms:
            fixel_file = os.path.join(fixel_path, sub, 'msmt_csd')
            if not os.path.exists(os.path.join(fixel_path, sub, 'msmt_csd')):
                os.mkdir(os.path.join(fixel_path, sub))
                os.mkdir(os.path.join(fixel_path, sub, 'msmt_csd'))
        else:
            fixel_file = os.path.join(fixel_path, sub, 'ss3t_csd')
            if not os.path.exists(os.path.join(fixel_path, sub, 'ss3t_csd')):
#                 os.mkdir(os.path.join(fixel_path, sub))
                os.mkdir(os.path.join(fixel_path, sub, 'ss3t_csd'))

        subprocess.call(['fod2fixel', mif_file, fixel_file, '-afd', 'afd.mif', '-disp', 'disp.mif',                          '-peak_amp', 'peak.mif', '-force'])
        clear_output(wait=True)
        sub_ctr += 1

def pred_nii_to_fixel(sub_list, nii_path, mif_path, fixel_path):
    if not os.path.exists(mif_path):
        os.mkdir(mif_path)
    if not os.path.exists(fixel_path):
        os.mkdir(fixel_path)

    sub_ctr=1
    for sub in sub_list:
        print('process: {}, progress:{}/{}'.format(sub, sub_ctr, len(sub_list)))

        nii = os.path.join(nii_path, sub + '.nii.gz')
        mif_file = os.path.join(mif_path, sub + '.mif.gz')

        subprocess.call(['mrconvert', nii, mif_file, '-force'])

        fixel_file = os.path.join(fixel_path, sub)

        subprocess.call(['fod2fixel', mif_file, fixel_file, '-afd', 'afd.mif', '-disp', 'disp.mif', '-peak_amp', 'peak.mif', '-force'])
        clear_output(wait=True)
        sub_ctr += 1
#         break




# In[44]:
# nii_path = '/home/data/HCP_SR/HCP_50/test/'
# mif_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_50_test_mif/'
# fixel_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_50_test_fixel/'
#
# nii_to_fixel(sub_list, nii_path, mif_path, fixel_path, ms=True)


# In[46]:
# nii_path = '/home/data/HCP_SR/HCP_SR_results/fodnet/'
# mif_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_SR_results_fodnet_mif/'
# fixel_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_SR_results_fodnet_fixel/'
#
# nii_to_fixel(sub_list, nii_path, mif_path, fixel_path, ms=True)


# In[60]:
#
# for sub in sub_list:
#
#     index_gt_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_50_test_fixel/{}/msmt_csd/index.mif'.format(sub)
#     direcions_gt_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_50_test_fixel/{}/msmt_csd/directions.mif'.format(sub)
#
#     index_est_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_SR_results_fodnet_fixel/{}/msmt_csd/index.mif'.format(sub)
#     directions_est_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_SR_results_fodnet_fixel/{}/msmt_csd/directions.mif'.format(sub)
#
#     output_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_SR_results_fodnet_msmt_csd_mae/{}'.format(sub)
#
#     mean_angular_error(index_gt_path, direcions_gt_path, index_est_path,                    directions_est_path, output_path, max_num_fixel_per_voxel=3)
# #     break

def compute_pred_fixel_metrics(sub_list, fixel_path, fixel_gt_path, output_path, model=None):
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    sub_ctr=1
    for sub in sub_list:
        print('process: {}, progress:{}/{}'.format(sub, sub_ctr, len(sub_list)))
        if model == 'fodnet':
            index_est_path = fixel_path + '/{}/msmt_csd/index.mif'.format(sub)
            directions_est_path = fixel_path + '/{}/msmt_csd/directions.mif'.format(sub)
            afd_est_path = fixel_path + '/{}/msmt_csd/afd.mif'.format(sub)
            peak_est_path = fixel_path + '/{}/msmt_csd/peak.mif'.format(sub)
        elif model == 'ss3t':
            index_est_path = fixel_path + '/{}/ss3t_csd/index.mif'.format(sub)
            directions_est_path = fixel_path + '/{}/ss3t_csd/directions.mif'.format(sub)
            afd_est_path = fixel_path + '/{}/ss3t_csd/afd.mif'.format(sub)
            peak_est_path = fixel_path + '/{}/ss3t_csd/peak.mif'.format(sub)
        else:
            index_est_path = fixel_path + '/{}/index.mif'.format(sub)
            directions_est_path = fixel_path + '/{}/directions.mif'.format(sub)
            afd_est_path = fixel_path + '/{}/afd.mif'.format(sub)
            peak_est_path = fixel_path + '/{}/peak.mif'.format(sub)

        index_gt_path = fixel_gt_path + '/{}/msmt_csd/index.mif'.format(sub)
        direcions_gt_path = fixel_gt_path + '/{}/msmt_csd/directions.mif'.format(sub)
        afd_gt_path = fixel_gt_path + '/{}/msmt_csd/afd.mif'.format(sub)
        peak_gt_path = fixel_gt_path + '/{}/msmt_csd/peak.mif'.format(sub)

        # index_est_path = fixel_path + '/{}/index.mif'.format(sub)
        # directions_est_path = fixel_path + '/{}/directions.mif'.format(sub)
        # afd_est_path = fixel_path + '/{}/afd.mif'.format(sub)
        # peak_est_path = fixel_path + '/{}/peak.mif'.format(sub)

        sub_output_path = output_path + '/{}'.format(sub)

        afd_peak(index_gt_path, direcions_gt_path, afd_gt_path, peak_gt_path, index_est_path, \
                 directions_est_path, afd_est_path, peak_est_path, sub_output_path, max_num_fixel_per_voxel=3)
        sub_ctr += 1
        # mean_angular_error(index_gt_path, direcions_gt_path, index_est_path, \
        #                    directions_est_path, output_path, max_num_fixel_per_voxel=3)
#     break


def mean_mae(sub_list, mae_path, brain_mask_path='/home/xinyi/fixel_based_analysis/HCP_SR/fodnet_brain_mask'):
    # In[182]:
    if os.path.exists(os.path.join(mae_path, 'mae_summary.txt')):
        subprocess.call(['rm', os.path.join(mae_path, 'mae_summary.txt')])
    summary_file = open(os.path.join(mae_path, 'mae_summary.txt'), 'w')

    entry = 'Compute MAE from' + mae_path + '\n'

    mean = []
    mean_est_mask = []
    min = []
    max = []
    for sub in sub_list:
        entry += ('case: ' + sub + '\n')
        print('case: ' + sub )

        mae_pred = load_mrtrix(mae_path + '/{}/mae.mif'.format(sub))

        est_mask = load_mrtrix(mae_path + '/{}/est_mask.mif'.format(sub))

        brain_mask, _ = load_fod_sh('/home/xinyi/fixel_based_analysis/HCP_SR/fodnet_brain_mask/{}_brain_mask.nii.gz'.format(sub))


        mean.append((np.sum(mae_pred.data*brain_mask[:,:,:,0])/np.sum(brain_mask[:,:,:,0])))
        mean_est_mask.append(np.sum(mae_pred.data[est_mask.data == 1]) / np.sum(est_mask.data))
        min.append(np.min(mae_pred.data))
        max.append(np.max(mae_pred.data))

        entry += ('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6]
                  + ' mean_est_mask: ' + str(mean_est_mask[-1])[:6] + '\n')
        print('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6]
              + ' mean_est_mask: ' + str(mean_est_mask[-1])[:6])
    #     if sub == '147737':
    #         print(mae_pred.shape, est_mask.shape)
    # #         print(np.sum(mae_pred), np.sum(est_mask))
    #         print(np.mean(mae_pred.data))
    #         print(np.sum(mae_pred.data[est_mask.data==1])/np.sum(est_mask.data))
    #         gt_fod_sh, gt_fod_affine = load_fod_sh('/home/data/HCP_SR/HCP_50/test/{}/msmt_csd/WM_FODs_normalised.nii.gz'.format(sub))
    #         brain_mask = gt_fod_sh.copy()[:,:,:,0]
    #         brain_mask[brain_mask==0]=0
    #         brain_mask[brain_mask!=0]=1
    #         print('147737 mae:', np.sum(mae_pred.data)/np.sum(brain_mask))

    mean_mean_info = 'mean of all subjects: {:.6f}±{:.6f}'.format(np.mean(mean), np.std(mean))
    mean_est_mask_info = 'mean_est_mask of all subjects: {:.6f}±{:.6f}'.format(np.mean(mean_est_mask), np.std(mean_est_mask))
    mean_min_info = 'min of all subjects: {:.6f}±{:.6f}'.format(np.mean(min), np.std(min))
    mean_max_info = 'max of all subjects: {:.6f}±{:.6f}'.format(np.mean(max), np.std(max))
    print(mean_mean_info)
    print(mean_est_mask_info)
    print(mean_min_info)
    print(mean_max_info)

    #
    # print('For record keeping: \n {:.4f}±{:.4f} {:.4f}±{:.4f} {:.4f}±{:.4f}' \
    #       .format(np.mean(mse_lst), np.std(mse_lst), np.mean(mae_lst), np.std(mae_lst), \
    #               np.mean(psnr), np.std(psnr)))
    #
    entry += (mean_mean_info + '\n')
    entry += (mean_est_mask_info + '\n')
    entry += (mean_min_info + '\n')
    entry += (mean_max_info + '\n')
    entry += '\n'

    summary_file.write(entry)
    summary_file.close()


def mean_peak(sub_list, peak_path, brain_mask_path='/home/xinyi/fixel_based_analysis/HCP_SR/fodnet_brain_mask'):
    # In[182]:
    if os.path.exists(os.path.join(peak_path, 'peak_summary.txt')):
        subprocess.call(['rm', os.path.join(peak_path, 'peak_summary.txt')])
    summary_file = open(os.path.join(peak_path, 'peak_summary.txt'), 'w')

    entry = 'Compute PEAK error from' + peak_path + '\n'

    mean = []
    mean_est_mask = []
    min = []
    max = []
    for sub in sub_list:
        entry += ('case: ' + sub + '\n')
        print('case: ' + sub )

        peak_pred = load_mrtrix(peak_path + '/{}/peak_error.mif'.format(sub))

        est_mask = load_mrtrix(peak_path + '/{}/est_mask.mif'.format(sub))

        brain_mask, _ = load_fod_sh('/home/xinyi/fixel_based_analysis/HCP_SR/fodnet_brain_mask/{}_brain_mask.nii.gz'.format(sub))


        mean.append((np.sum(peak_pred.data*brain_mask[:,:,:,0])/np.sum(brain_mask[:,:,:,0])))
        mean_est_mask.append(np.sum(peak_pred.data[est_mask.data == 1]) / np.sum(est_mask.data))
        min.append(np.min(peak_pred.data))
        max.append(np.max(peak_pred.data))

        entry += ('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6]
                  + ' mean_est_mask: ' + str(mean_est_mask[-1])[:6] + '\n')
        print('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6]
              + ' mean_est_mask: ' + str(mean_est_mask[-1])[:6] + '\n')

    mean_mean_info = 'mean of all subjects: {:.6f}±{:.6f}'.format(np.mean(mean), np.std(mean))
    mean_est_mask_info = 'mean_est_mask of all subjects: {:.6f}±{:.6f}'.format(np.mean(mean_est_mask),
                                                                               np.std(mean_est_mask))
    mean_min_info = 'min of all subjects: {:.6f}±{:.6f}'.format(np.mean(min), np.std(min))
    mean_max_info = 'max of all subjects: {:.6f}±{:.6f}'.format(np.mean(max), np.std(max))
    print(mean_mean_info)
    print(mean_est_mask_info)
    print(mean_min_info)
    print(mean_max_info)

    entry += (mean_mean_info + '\n')
    entry += (mean_est_mask_info + '\n')
    entry += (mean_min_info + '\n')
    entry += (mean_max_info + '\n')
    entry += '\n'

    summary_file.write(entry)
    summary_file.close()

def mean_afd(sub_list, peak_path, brain_mask_path='/home/xinyi/fixel_based_analysis/HCP_SR/fodnet_brain_mask'):
    # In[182]:
    if os.path.exists(os.path.join(peak_path, 'afd_summary.txt')):
        subprocess.call(['rm', os.path.join(peak_path, 'afd_summary.txt')])
    summary_file = open(os.path.join(peak_path, 'afd_summary.txt'), 'w')

    entry = 'Compute AFD error from' + peak_path + '\n'

    mean = []
    mean_est_mask = []
    min = []
    max = []
    for sub in sub_list:
        entry += ('case: ' + sub + '\n')
        print('case: ' + sub )

        peak_pred = load_mrtrix(peak_path + '/{}/afd_error.mif'.format(sub))

        est_mask = load_mrtrix(peak_path + '/{}/est_mask.mif'.format(sub))

        brain_mask, _ = load_fod_sh('/home/xinyi/fixel_based_analysis/HCP_SR/fodnet_brain_mask/{}_brain_mask.nii.gz'.format(sub))


        mean.append((np.sum(peak_pred.data*brain_mask[:,:,:,0])/np.sum(brain_mask[:,:,:,0])))
        mean_est_mask.append(np.sum(peak_pred.data[est_mask.data == 1]) / np.sum(est_mask.data))
        min.append(np.min(peak_pred.data))
        max.append(np.max(peak_pred.data))

        entry += ('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6]
                  + ' mean_est_mask: ' + str(mean_est_mask[-1])[:6] + '\n')
        print('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6]
              + ' mean_est_mask: ' + str(mean_est_mask[-1])[:6] + '\n')

    mean_mean_info = 'mean of all subjects: {:.6f}±{:.6f}'.format(np.mean(mean), np.std(mean))
    mean_est_mask_info = 'mean_est_mask of all subjects: {:.6f}±{:.6f}'.format(np.mean(mean_est_mask),
                                                                               np.std(mean_est_mask))
    mean_min_info = 'min of all subjects: {:.6f}±{:.6f}'.format(np.mean(min), np.std(min))
    mean_max_info = 'max of all subjects: {:.6f}±{:.6f}'.format(np.mean(max), np.std(max))
    print(mean_mean_info)
    print(mean_est_mask_info)
    print(mean_min_info)
    print(mean_max_info)

    entry += (mean_mean_info + '\n')
    entry += (mean_est_mask_info + '\n')
    entry += (mean_min_info + '\n')
    entry += (mean_max_info + '\n')
    entry += '\n'

    summary_file.write(entry)
    summary_file.close()

def compute_acc(sub_list, pred_fod_path, gt_fod_path, output_path, model=None):

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    sub_ctr=1
    for sub in sub_list:
        print('process: {}, progress:{}/{}'.format(sub, sub_ctr, len(sub_list)))
        if model == 'fodnet':
            pred_fod_sh, pred_fod_affine = load_fod_sh(os.path.join(pred_fod_path, sub, 'msmt_csd', 'SR_WM_FODs_normalised.nii.gz'))
        elif model == 'ss3t':
            pred_fod_sh, pred_fod_affine = load_fod_sh(os.path.join(pred_fod_path, sub, 'ss3t_csd', 'WM_FODs_normalised.nii.gz'))
        else:
            pred_fod_sh, pred_fod_affine = load_fod_sh(os.path.join(pred_fod_path, sub+'.nii.gz'))
        print('pred_fod_sh', pred_fod_sh.shape)
        gt_fod_sh, gt_fod_affine = load_fod_sh(os.path.join(gt_fod_path, sub, 'msmt_csd', 'WM_FODs_normalised.nii.gz'))
        print('gt_fod_sh', gt_fod_sh.shape)

    # index_gt.data = mae
    # save_mrtrix(mae_path, index_gt)
        sub_output_path = os.path.join(output_path, sub, 'acc.nii.gz')
        if not os.path.exists(os.path.join(output_path, sub)):
            os.mkdir(os.path.join(output_path, sub))
        acc = angular_corr(pred_fod_sh, gt_fod_sh)

        # Convert numpy array to NIFTI
        nifti = nib.Nifti1Image(acc, affine=pred_fod_affine)
        # Save segmentation to disk
        nib.save(nifti, sub_output_path)
        sub_ctr += 1

    # # print(acc[52,102,76], acc.shape)
    # print(np.nanmax(acc), np.nanmin(acc))
    #
    # brain_mask, _ = load_fod_sh()

# In[146]:


# # angular correlation coefficient
# pred_fod_sh, pred_fod_affine = load_fod_sh('/home/xinyi/fodnet_published/dataset/111009/fodnet/SR_WM_FODs_normalised.nii.gz')
# print('pred_fod_sh', pred_fod_sh.shape)
# gt_fod_sh, gt_fod_affine = load_fod_sh('/home/xinyi/fodnet_published/dataset/111009/msmt_csd/WM_FODs_normalised.nii.gz')
# print('gt_fod_sh', gt_fod_sh.shape)
# ss3t_fod_sh, ss3t_fod_affine = load_fod_sh('/home/xinyi/fodnet_published/dataset/111009/ss3t_csd/WM_FODs_normalised.nii.gz')
#
#
# # Suppress/hide the warning
# np.seterr(invalid='ignore')
# acc = angular_corr(pred_fod_sh, gt_fod_sh)
# # print(acc[52,102,76], acc.shape)
# print(np.nanmax(acc), np.nanmin(acc))
#
# ss3t_acc = angular_corr(ss3t_fod_sh, gt_fod_sh)
# # print(ss3t_acc[52,102,76], ss3t_acc.shape)
# print(np.nanmax(ss3t_acc), np.nanmin(ss3t_acc), np.nanmean(ss3t_acc))
#

def mean_acc(sub_list, acc_path, brain_mask_path='/home/xinyi/fixel_based_analysis/HCP_SR/fodnet_brain_mask'):
    # In[182]:
    if os.path.exists(os.path.join(acc_path, 'acc_summary.txt')):
        subprocess.call(['rm', os.path.join(acc_path, 'acc_summary.txt')])
    summary_file = open(os.path.join(acc_path, 'acc_summary.txt'), 'w')

    entry = 'Compute ACC from' + acc_path + '\n'

    mean = []
    min = []
    max = []
    upper_quartile = []
    lower_quartile = []
    for sub in sub_list:
        entry += ('case: ' + sub + '\n')
        print('case: ' + sub)

        acc, _ = load_fod_sh(os.path.join(acc_path, sub, 'acc.nii.gz'))
        mask, _ = load_fod_sh(os.path.join(brain_mask_path, '{}_brain_mask.nii.gz'.format(sub)))

        mask = mask[:,:,:,0]

        # # print('brain mask', np.nanmax(acc*brain_mask), np.nanmin(acc*brain_mask), np.nanmean(acc*brain_mask))
        #
        mean.append((np.nansum(acc*mask)/np.sum(mask)))
        min.append(np.nanmin(acc*mask))
        max.append(np.nanmax(acc*mask))
        lower_quartile.append(np.nanquantile(acc*mask, 0.25))
        upper_quartile.append(np.nanquantile(acc*mask, 0.75))

        entry += ('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6] + \
                  ' lower_quartile: ' + str(lower_quartile[-1])[:6] + ' upper_quartile: ' + str(upper_quartile[-1])[:6] + '\n')
        print('mean: ' + str(mean[-1])[:6] + ' min: ' + str(min[-1])[:6] + ' max: ' + str(max[-1])[:6] + \
              ' lower_quartile: ' + str(lower_quartile[-1])[:6] + ' upper_quartile: ' + str(upper_quartile[-1])[:6])

    mean_mean_info = 'mean of all subjects: {:.6f}±{:.6f}'.format(np.mean(mean), np.std(mean))
    mean_min_info = 'min of all subjects: {:.6f}±{:.6f}'.format(np.mean(min), np.std(min))
    mean_max_info = 'max of all subjects: {:.6f}±{:.6f}'.format(np.mean(max), np.std(max))
    mean_upper_info = 'upper quartile of all subjects: {:.6f}±{:.6f}'.format(np.mean(upper_quartile), np.std(upper_quartile))
    mean_lower_info = 'lower quartile of all subjects: {:.6f}±{:.6f}'.format(np.mean(lower_quartile), np.std(lower_quartile))
    print(mean_mean_info)
    print(mean_min_info)
    print(mean_max_info)
    print(mean_upper_info)
    print(mean_lower_info)

    entry += (mean_mean_info + '\n')
    entry += (mean_min_info + '\n')
    entry += (mean_max_info + '\n')
    entry += (mean_upper_info + '\n')
    entry += (mean_lower_info + '\n')
    entry += '\n'

    summary_file.write(entry)
    summary_file.close()
# # brain_mask = nib.load('/home/xinyi/fodnet_published/dataset/111009/brain_mask.nii.gz')
# # brain_mask = brain_mask.get_fdata()
# # brain_mask = gt_fod_sh.copy()[:,:,:,0]
# # brain_mask[brain_mask!=0]=1
# # print((acc*brain_mask).shape)
# # print('brain mask', np.nanmax(acc*brain_mask), np.nanmin(acc*brain_mask), np.nanmean(acc*brain_mask))
#
#
# # In[147]:
#
#
# tt_mask=nib.load('/home/xinyi/fodnet_published/dataset/111009/fsl_5ttgen.nii.gz').get_fdata()
# print(tt_mask.shape)
#
# tt_mask = tt_mask.copy()
# # tt_mask[tt_mask>0.05]=1
# # tt_mask[tt_mask<0.1]=0
# tt_mask1=tt_mask[:,:,:,0] # CGM
# tt_mask2=tt_mask[:,:,:,1] # SGM
# tt_mask3=tt_mask[:,:,:,2] # WM
# tt_mask4=tt_mask[:,:,:,3] # CSF
# tt_mask5=tt_mask[:,:,:,4]
#
# print(np.nansum(acc*tt_mask1)/np.nansum(tt_mask1))
# print(np.nanmin(acc))
#
# print('ss3t_acc WM&CGM', np.nanmax(ss3t_acc*(tt_mask1+tt_mask3)), np.nanmin(ss3t_acc*(tt_mask1+tt_mask3)), np.nansum(ss3t_acc*(tt_mask1+tt_mask3))/np.sum(tt_mask1+tt_mask3), np.nanmean(ss3t_acc*(tt_mask1+tt_mask3)))
# print('ss3t_acc WM&SGM', np.nanmax(ss3t_acc*(tt_mask1+tt_mask2)), np.nanmin(ss3t_acc*(tt_mask1+tt_mask2)), np.nansum(ss3t_acc*(tt_mask1+tt_mask2))/np.sum(tt_mask1+tt_mask2), np.nanmean(ss3t_acc*(tt_mask1+tt_mask2)))
# print('ss3t_acc WM', np.nanmax(ss3t_acc*tt_mask3), np.nanmin(ss3t_acc*tt_mask3), np.nansum(ss3t_acc*tt_mask3)/np.sum(tt_mask3), np.nanmean(ss3t_acc*tt_mask3))
# print('ss3t_acc tt mask4', np.nanmax(ss3t_acc*tt_mask4), np.nanmin(ss3t_acc*tt_mask4), np.nansum(ss3t_acc*tt_mask4)/np.sum(tt_mask4), np.nanmean(ss3t_acc*tt_mask4))
# print('ss3t_acc tt mask5', np.nanmax(ss3t_acc*tt_mask5), np.nanmin(ss3t_acc*tt_mask5), np.nansum(ss3t_acc*tt_mask4)/np.sum(tt_mask4), np.nanmean(ss3t_acc*tt_mask5))
#
# ss3t_acc = acc
# print('fodnet WM&CGM', np.nanmax(ss3t_acc*(tt_mask1+tt_mask3)), np.nanmin(ss3t_acc*(tt_mask1+tt_mask3)), np.nansum(ss3t_acc*(tt_mask1+tt_mask3))/np.sum(tt_mask1+tt_mask3), np.nanmean(ss3t_acc*(tt_mask1+tt_mask3)))
# print('fodnet WM&SGM', np.nanmax(ss3t_acc*(tt_mask1+tt_mask2)), np.nanmin(ss3t_acc*(tt_mask1+tt_mask2)), np.nansum(ss3t_acc*(tt_mask1+tt_mask2))/np.sum(tt_mask1+tt_mask2), np.nanmean(ss3t_acc*(tt_mask1+tt_mask2)))
# print('fodnet WM', np.nanmax(ss3t_acc*tt_mask3), np.nanmin(ss3t_acc*tt_mask3), np.nansum(ss3t_acc*tt_mask3)/np.sum(tt_mask3), np.nanmean(ss3t_acc*tt_mask3))
# print('fodnet tt mask4', np.nanmax(ss3t_acc*tt_mask4), np.nanmin(ss3t_acc*tt_mask4), np.nansum(ss3t_acc*tt_mask4)/np.sum(tt_mask4), np.nanmean(ss3t_acc*tt_mask4))
# print('fodnet tt mask5', np.nanmax(ss3t_acc*tt_mask5), np.nanmin(ss3t_acc*tt_mask5), np.nansum(ss3t_acc*tt_mask4)/np.sum(tt_mask4), np.nanmean(ss3t_acc*tt_mask5))
#
#
#
#
# print(fod_sh.shape)
#
#
# # In[151]:
#
#
# for sub in sub_list:
#     if sub == '147737':
#         pred_fixel_dir_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_50_test_fixel/{}/ss3t_csd/'.format(sub)
#         gt_fixel_dir_path = '/home/xinyi/fixel_based_analysis/HCP_SR/HCP_50_test_fixel/{}/msmt_csd/'.format(sub)
#
#         gt_fod_sh, gt_fod_affine = load_fod_sh('/home/data/HCP_SR/HCP_50/test/{}/msmt_csd/WM_FODs_normalised.nii.gz'.format(sub))
# #         print('gt_fod_sh', gt_fod_sh.shape)
#         brain_mask = gt_fod_sh.copy()[:,:,:,0]
#         brain_mask[brain_mask!=0]=1
#
#         fixelmetric = evaluation_fixel_mask(pred_fixel_dir_path, gt_fixel_dir_path, anatomical_mask=brain_mask, num_fixel=1)


# In[154]:


# fixelmetric.avg_peak
# fixelmetric.avg_afd
# fixelmetric.avg_disp
# fixelmetric.avg_angle


# In[ ]: