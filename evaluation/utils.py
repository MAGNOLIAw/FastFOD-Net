import os
import re
import nibabel as nib
from mrtrix import load_mrtrix

def find_file(name, dirname):
    """
    Find a file within a directory matching the given name.
    """
    result = list(filter(
        lambda x: not os.path.isdir(x) and re.search(name, x),
        os.listdir(dirname)
    ))
    if len(result) > 1:
        result.pop(0)
    return os.path.join(dirname, result[0]) if result else None

def load_image(image_path):
    """
    Load an image from a file path.
    """
    if image_path.endswith(('.mif.gz', '.mif')):
        return load_mrtrix(image_path).data
    elif image_path.endswith(('.nii.gz', '.nii')):
        return nib.load(image_path).get_fdata()
    else:
        raise IOError(f'File extension not supported: {image_path}')

def load_fixel_data(fixel_path, directions='directions.mif', index='index.mif', afd='afd.mif', peak='peak.mif'):
    """
    Load fixel data from a given directory.
    """
    index_tuples = load_mrtrix(os.path.join(fixel_path, index)).data
    afd_vector = load_mrtrix(os.path.join(fixel_path, afd)).data.squeeze()
    peak_vector = load_mrtrix(os.path.join(fixel_path, peak)).data.squeeze()
    dir_matrix = load_mrtrix(os.path.join(fixel_path, directions)).data.squeeze()
    
    return index_tuples, afd_vector, peak_vector, dir_matrix
