import os
import glob
import nibabel as nib
from scipy import ndimage
import numpy as np


def read_nifti_file(filepath):
    """
    Read and load volume
    """
    # Read file
    scan = nib.load(filepath)
    # Get raw data
    scan = scan.get_fdata()
    return scan


def normalize(volume):
    """
    Normalize the volume
    """
    min, max = -1000, 400
    volume[volume < min] = min
    volume[volume > max] = max
    volume = ((volume - min) / (max - min)).astype("float32")
    return volume


def resize_volume(img, d=64, w=64, h=64):
    """
    Resize across z-axis
    """
    # Get current depth
    current_depth, current_width, current_height = img.shape[-1], img.shape[0], img.shape[1]
    depth, width, height = current_depth / d, current_width / w, current_height / h
    # Compute depth factor
    depth_f, width_f, height_f = 1 / depth, 1 / width, 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_f, height_f, depth_f), order=1)
    return img


def process_scan(path):
    """
    Read and resize volume
    """
    volume = read_nifti_file(path)  # Read scan
    volume = normalize(volume)      # Normalize
    volume = resize_volume(volume)  # Resize width, height and depth
    return volume


def datasets(target: str='age'):
    path_unmeddep = os.path.join(os.getcwd(), f'data.unmeddep/{target}/')
    scans_unmeddep = glob.glob(path_unmeddep + '*.nii.gz', recursive=False)
    scans_unmeddep = np.array([process_scan(path) for path in scans_unmeddep])
    labels_unmeddep = np.array([1 for _ in range(len(scans_unmeddep))])
    
    path_other = os.path.join(os.getcwd(), 'data.other/age/')
    scans_other = glob.glob(path_other + '*.nii.gz', recursive=False)
    scans_other = np.array([process_scan(path) for path in scans_other])
    labels_other = np.array([1 for _ in range(len(scans_other))])

    return scans_unmeddep, labels_unmeddep, scans_other, labels_other


