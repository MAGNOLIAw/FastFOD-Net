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
import numpy
import math
import torch
import os

#-----------------------------------------------------#
#                 Image Sample - class                #
#-----------------------------------------------------#
# Object containing an image and the associated segmentation
class HCP_FOD_Sample:
    # Initialize class variable
    index = None
    img_data = None
    seg_data = None
    pred_data = None
    shape = None
    channels = None
    classes = None
    details = None

    # Create a Sample object
    def __init__(self, index, image, channels, classes, info=None):
        # Cache data
        self.index = index
        self.img_data = image
        self.channels = channels
        self.classes = classes
        self.shape = self.img_data.shape
        # self.aff = aff # affine for nifti
        self.info = info  # affine for nifti


    # Add and preprocess a segmentation annotation
    def add_segmentation(self, seg):
        self.seg_data = seg

    # Add and preprocess a ground truth
    def add_gt(self, gt):
        self.gt_data = gt

    # Add and preprocess a prediction annotation
    def add_prediction(self, pred):
        self.pred_data = pred

    # Add optional information / details for custom usage
    def add_details(self, details):
        self.details = details

