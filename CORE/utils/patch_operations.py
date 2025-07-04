#==============================================================================#  
#  Authors:     Xinyi Wang                                                     #
#  License:     GNU GPL v3.0                                                   #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
#External libraries
import numpy as np
import math
from batchgenerators.augmentations.utils import pad_nd_image

import torch.nn.functional as F

#-----------------------------------------------------#
#      Pad and crop patch to desired patch shape      #
#-----------------------------------------------------#
def pad_patch(patch, patch_shape, return_slicer=False):
    """
    Pad a patch to the target patch shape using minimum padding.
    Converts channel-last to channel-first format for compatibility.

    Args:
        patch (np.ndarray): Input patch with shape (H, W, D, C).
        patch_shape (tuple): Target patch shape.
        return_slicer (bool): Whether to return the slicer for cropping.

    Returns:
        np.ndarray or (np.ndarray, slicer): Padded patch and optionally the slicer.
    """
    # Initialize stat length to overwrite batchgenerators default
    kwargs = {"stat_length": None}
    # Transform prediction from channel-last to channel-first structure
    patch = np.moveaxis(patch, -1, 1)
    # Run padding
    padding_results = pad_nd_image(patch, new_shape=patch_shape,
                                   mode="minimum", return_slicer=return_slicer,
                                   kwargs=kwargs)
    # Return padding results
    if return_slicer:
        # Transform data from channel-first back to channel-last structure
        padded_patch = np.moveaxis(padding_results[0], 1, -1)
        return padded_patch, padding_results[1]
    else:
        # Transform data from channel-first back to channel-last structure
        padding_results = np.moveaxis(padding_results, 1, -1)
        return padding_results

def crop_patch(patch, slicer):
    """
    Crop a padded patch using a slicer.

    Args:
        patch (np.ndarray): Padded patch with shape (H, W, D, C).
        slicer (list): List of slice objects.

    Returns:
        np.ndarray: Cropped patch.
    """
    # Transform prediction from channel-last to channel-first structure
    patch = np.moveaxis(patch, -1, 1)
    # Exclude the number of batches and classes from the slice range
    slicer[0] = slice(None)
    slicer[1] = slice(None)
    # Crop patches according to slicer
    patch_cropped = patch[tuple(slicer)]
    # Transform data from channel-first back to channel-last structure
    patch_cropped = np.moveaxis(patch_cropped, 1, -1)
    # Return cropped patch
    return patch_cropped

#-----------------------------------------------------#
#         Slice and Concatenate Function Hubs         #
#-----------------------------------------------------#
# Slice a matrix
def slice_matrix(array, window, overlap, three_dim, index=None, save_coords=False):
    """
    Dispatcher function to slice 2D or 3D matrices.

    Args:
        array (np.ndarray): Input matrix.
        window (tuple): Patch size.
        overlap (tuple): Overlap size.
        three_dim (bool): Whether to use 3D slicing.
        index (optional): Subject index for coordinate tracking.
        save_coords (bool): Whether to save coordinates.

    Returns:
        list: Sliced patches.
    """
    if three_dim: return slice_3Dmatrix(array, window, overlap, index, save_coords)
    else: return slice_2Dmatrix(array, window, overlap)

# Concatenate a matrix
def concat_matrices(patches, image_size, window, overlap, three_dim, coords=None):
    """
    Dispatcher function to concatenate 2D or 3D patches.

    Args:
        patches (list): List of image patches.
        image_size (tuple): Final image size.
        window (tuple): Patch size.
        overlap (tuple): Overlap size.
        three_dim (bool): Whether to use 3D concatenation.
        coords (list): List of coordinate dictionaries.

    Returns:
        np.ndarray: Reconstructed image.
    """
    if three_dim: return concat_3Dmatrices(patches, image_size, window, overlap, coords)
    else: return concat_2Dmatrices(patches, image_size, window, overlap)

#-----------------------------------------------------#
#          Slice and Concatenate 2D Matrices          #
#-----------------------------------------------------#
# Slice a 2D matrix
def slice_2Dmatrix(array, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) /
                            float(window[1] - overlap[1])))

    # Iterate over it x,y
    patches = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):

            # Define window edges
            x_start = x*window[0] - x*overlap[0]
            x_end = x_start + window[0]
            y_start = y*window[1] - y*overlap[1]
            y_end = y_start + window[1]
            # Adjust ends
            if(x_end > len(array)):
                # Create an overlapping patch for the last images / edges
                # to ensure the fixed patch/window sizes
                x_start = len(array) - window[0]
                x_end = len(array)
                # Fix for MRIs which are smaller than patch size
                if x_start < 0 : x_start = 0
            if(y_end > len(array[0])):
                y_start = len(array[0]) - window[1]
                y_end = len(array[0])
                # Fix for MRIs which are smaller than patch size
                if y_start < 0 : y_start = 0
            # Cut window
            window_cut = array[x_start:x_end,y_start:y_end]
            # Add to result list
            patches.append(window_cut)
    return patches

# Concatenate a list of patches together to a numpy matrix
def concat_2Dmatrices(patches, image_size, window, overlap):
    # Calculate steps
    steps_x = int(math.ceil((image_size[0] - overlap[0]) /
                            float(window[0] - overlap[0])))
    steps_y = int(math.ceil((image_size[1] - overlap[1]) /
                            float(window[1] - overlap[1])))

    # Iterate over it x,y,z
    matrix_x = None
    matrix_y = None
    matrix_z = None
    pointer = 0
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            # Calculate pointer from 2D steps to 1D list of patches
            pointer = x*steps_y + y
            # Connect current tmp Matrix Z to tmp Matrix Y
            if y == 0:
                matrix_y = patches[pointer]
            else:
                matrix_p = patches[pointer]
                # Handle y-axis overlap
                slice_overlap = calculate_overlap(y, steps_y, overlap,
                                                  image_size, window, 1)
                matrix_y, matrix_p = handle_overlap(matrix_y, matrix_p,
                                                    slice_overlap,
                                                    axis=1)
                matrix_y = np.concatenate((matrix_y, matrix_p), axis=1)
        # Connect current tmp Matrix Y to final Matrix X
        if x == 0:
            matrix_x = matrix_y
        else:
            # Handle x-axis overlap
            slice_overlap = calculate_overlap(x, steps_x, overlap,
                                              image_size, window, 0)
            matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y,
                                                slice_overlap,
                                                axis=0)
            matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)
    # Return final combined matrix
    return(matrix_x)

#-----------------------------------------------------#
#          Slice and Concatenate 3D Matrices          #
#-----------------------------------------------------#
# Slice a 3D matrix
def slice_3Dmatrix(array, window, overlap, index=None, save_coords=False):
    """
    Slice a 3D matrix into overlapping patches.

    Args:
        array (np.ndarray): Input 3D array.
        window (tuple): Patch size (Dx, Dy, Dz).
        overlap (tuple): Overlap size (Dx, Dy, Dz).
        index (int, optional): Identifier for the subject/sample.
        save_coords (bool): Whether to save coordinate information.

    Returns:
        list: List of 3D patches.
        (optional) list: Corresponding coordinate dictionaries.
    """
    # Calculate steps
    steps_x = int(math.ceil((len(array) - overlap[0]) / float(window[0] - overlap[0])))
    steps_y = int(math.ceil((len(array[0]) - overlap[1]) / float(window[1] - overlap[1])))
    steps_z = int(math.ceil((len(array[0][0]) - overlap[2]) / float(window[2] - overlap[2])))

    # Iterate over it x,y,z
    patches = []
    coords = []
    for x in range(0, steps_x):
        for y in range(0, steps_y):
            for z in range(0, steps_z):
                # Define window edges
                x_start = x*window[0] - x*overlap[0]
                x_end = x_start + window[0]
                y_start = y*window[1] - y*overlap[1]
                y_end = y_start + window[1]
                z_start = z*window[2] - z*overlap[2]
                z_end = z_start + window[2]

                # Adjust ends
                if(x_end > len(array)):
                    # Create an overlapping patch for the last images / edges
                    # to ensure the fixed patch/window sizes
                    x_start = len(array) - window[0]
                    x_end = len(array)
                    # Fix for MRIs which are smaller than patch size
                    if x_start < 0 : x_start = 0
                if(y_end > len(array[0])):
                    y_start = len(array[0]) - window[1]
                    y_end = len(array[0])
                    # Fix for MRIs which are smaller than patch size
                    if y_start < 0 : y_start = 0
                if(z_end > len(array[0][0])):
                    z_start = len(array[0][0]) - window[2]
                    z_end = len(array[0][0])
                    # Fix for MRIs which are smaller than patch size
                    if z_start < 0 : z_start = 0

                # Cut window
                window_cut = array[x_start:x_end,y_start:y_end,z_start:z_end]
                # Add to result list
                patches.append(window_cut)

                if save_coords:
                    coords.append({
                        'index': index,
                        'x_start': x_start, 'x_end': x_end,
                        'y_start': y_start, 'y_end': y_end,
                        'z_start': z_start, 'z_end': z_end
                    })

    return (patches, coords) if save_coords else patches

# Concatenate a list of patches together to a numpy matrix
def concat_3Dmatrices(patches, image_size, window, overlap, coords=None):
    """
    Reconstruct a 3D matrix from overlapping patches.

    Args:
        patches (list): List of 3D patches.
        image_size (tuple): Final reconstructed image shape (Dx, Dy, Dz).
        window (tuple): Patch dimensions.
        overlap (tuple): Overlap size in each dimension.
        coords (list, optional): If provided, uses explicit coordinates for merging.

    Returns:
        np.ndarray: Reconstructed 3D image.
    """
    if coords is None:
        # Calculate steps
        steps_x = int(math.ceil((image_size[0] - overlap[0]) / float(window[0] - overlap[0])))
        steps_y = int(math.ceil((image_size[1] - overlap[1]) / float(window[1] - overlap[1])))
        steps_z = int(math.ceil((image_size[2] - overlap[2]) / float(window[2] - overlap[2])))

        matrix_x = None
        matrix_y = None
        matrix_z = None
        pointer = 0
        counts = np.zeros((steps_x, steps_y, steps_z))

        # Iterate over it x,y,z
        for x in range(0, steps_x):
            for y in range(0, steps_y):
                for z in range(0, steps_z):
                    # Calculate pointer from 3D steps to 1D list of patches
                    pointer = z + y*steps_z + x*steps_y*steps_z

                    # Connect current patch to temporary Matrix Z
                    if z == 0:
                        matrix_z = patches[pointer]
                    else:
                        matrix_p = patches[pointer]
                        # Handle z-axis overlap
                        counts[x, y, z] += 1
                        slice_overlap = calculate_overlap(z, steps_z, overlap, image_size, window, 2)
                        matrix_z, matrix_p = handle_overlap(matrix_z, matrix_p, slice_overlap, axis=2)
                        matrix_z = np.concatenate((matrix_z, matrix_p), axis=2)
                        
                # Connect current tmp Matrix Z to tmp Matrix Y
                if y == 0:
                    matrix_y = matrix_z
                else:
                    # Handle y-axis overlap
                    counts[x, y, z] += 1
                    slice_overlap = calculate_overlap(y, steps_y, overlap, image_size, window, 1)
                    matrix_y, matrix_z = handle_overlap(matrix_y, matrix_z, slice_overlap, axis=1)
                    matrix_y = np.concatenate((matrix_y, matrix_z), axis=1)

            # Connect current tmp Matrix Y to final Matrix X
            if x == 0:
                matrix_x = matrix_y
            else:
                # Handle x-axis overlap
                slice_overlap = calculate_overlap(x, steps_x, overlap, image_size, window, 0)
                matrix_x, matrix_y = handle_overlap(matrix_x, matrix_y, slice_overlap, axis=0)
                matrix_x = np.concatenate((matrix_x, matrix_y), axis=0)

        # Return final combined matrix
        return (matrix_x)
    
    else:
        img = np.zeros(image_size, dtype=np.float32)
        counts = np.zeros(image_size, dtype=np.float32)

        for ci, coord in enumerate(coords):
            sub_index = coord['index']
            x_start, x_end = coord['x_start'], coord['x_end']
            y_start, y_end = coord['y_start'], coord['y_end']
            z_start, z_end = coord['z_start'], coord['z_end']
            dataloader_idx = coord['dataloader_idx']
            # if dataloader_idx != ci:
            #     print('patches were shuffle, please be careful!')
            patch = patches[dataloader_idx, :, :, :]

            img[x_start:x_end, y_start:y_end, z_start:z_end] += patch
            counts[x_start:x_end, y_start:y_end, z_start:z_end] += 1

        counts[counts == 0] = 1
        img /= counts
        return img


#-----------------------------------------------------#
#          Subroutines for the Concatenation          #
#-----------------------------------------------------#
# Calculate the overlap of the current matrix slice
def calculate_overlap(pointer, steps, overlap, image_size, window, axis):
    """
    Calculate overlap size for boundary patches.

    Args:
        pointer (int): Current step index.
        steps (int): Total number of steps.
        overlap (tuple): Default overlap sizes.
        image_size (tuple): Final image size.
        window (tuple): Patch size.
        axis (int): Dimension along which to compute.

    Returns:
        int: Overlap size.
    """
    # Overlap: IF last axis-layer -> use special overlap size
    if pointer == steps-1 and not (image_size[axis]-overlap[axis]) \
                                    % (window[axis]-overlap[axis]) == 0:
        current_overlap = window[axis] - \
                            (image_size[axis] - overlap[axis]) % \
                            (window[axis] - overlap[axis])
    # Overlap: ELSE -> use default overlap size
    else:
        current_overlap = overlap[axis]
    # Return overlap
    return current_overlap

# Handle the overlap of two overlapping matrices
def handle_overlap(matrixA, matrixB, overlap, axis):
    """
    Merge two overlapping matrices by averaging overlapping regions.

    Args:
        matrixA (np.ndarray): First matrix.
        matrixB (np.ndarray): Second matrix.
        overlap (int): Overlap size.
        axis (int): Axis along which overlap occurs.

    Returns:
        tuple: Updated matrixA and matrixB after resolving overlap.
    """
    # Access overllaping slice from matrix A
    idxA = [slice(None)] * matrixA.ndim
    matrixA_shape = matrixA.shape
    idxA[axis] = range(matrixA_shape[axis] - overlap, matrixA_shape[axis])
    sliceA = matrixA[tuple(idxA)]
    # Access overllaping slice from matrix B
    idxB = [slice(None)] * matrixB.ndim
    idxB[axis] = range(0, overlap)
    sliceB = matrixB[tuple(idxB)]
    # Calculate Average prediction values between the two matrices
    # and save them in matrix A
    matrixA[tuple(idxA)] = np.mean(np.array([sliceA, sliceB]), axis=0)
    # Remove overlap from matrix B
    matrixB = np.delete(matrixB, [range(0, overlap)], axis=axis)
    # Return processed matrices
    return matrixA, matrixB


def find_bounding_box(brain):
    x, y, z = brain.shape[0], brain.shape[1], brain.shape[2]
    for i in range(z):
        slice = brain[:,:,i]
        if np.sum(slice) > 0:
            save_z_from_I = i
            break

    for i in reversed(range(z)):
        slice = brain[:,:,i]
        if np.sum(slice) > 0:
            save_z_from_S = i
            break

    for i in range(y):
        slice = brain[:, i, :]
        if np.sum(slice) > 0:
            save_y_from_P = i
            break

    for i in reversed(range(y)):
        slice = brain[:, i, :]
        if np.sum(slice) > 0:
            save_y_from_A = i
            break

    for i in range(x):
        slice = brain[i,:,:]
        if np.sum(slice) > 0:
            save_x_from_L = i
            break

    for i in reversed(range(x)):
        slice = brain[i,:,:]
        if np.sum(slice) > 0:
            save_x_from_R = i
            break

    return save_x_from_L, save_x_from_R, save_y_from_P, save_y_from_A, save_z_from_I, save_z_from_S


def get_pad(in_,  ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    pad = int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)
    return pad

def pad_to(x, stride):
    h, w, d = x.shape[-3:]
    # print(h,w,d)

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    if d % stride > 0:
        new_d = d + stride - d % stride
    else:
        new_d = d
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    ld, ud = int((new_d - d) / 2), int(new_d - d) - int((new_d - d) / 2)
    pads = (ld, ud, lw, uw, lh, uh)

    # zero-padding by default.
    # See others at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.pad
    out = F.pad(x, pads, "constant", 0)

    return out, pads

def unpad(x, pad):
    if pad[4]+pad[5] > 0:
        x = x[:,:,pad[4]:-pad[5],:,:]
    if pad[2]+pad[3] > 0:
        x = x[:,:,:,pad[2]:-pad[3],:]
    if pad[0]+pad[1] > 0:
        x = x[:,:,:,:,pad[0]:-pad[1]]
    return x


