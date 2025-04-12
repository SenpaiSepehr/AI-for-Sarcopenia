import os
import shutil
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import measure
import glob
import csv
from matplotlib import colors
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from scipy.interpolate import RegularGridInterpolator
import SimpleITK as sitk
from scipy.ndimage import zoom
from scipy.stats import pearsonr
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import warnings

# Suppress only RuntimeWarnings

def preprocess_CT_volume(sitk_image, target_spacing=1, mode='frontal',threshold = True):

    """Performs preprocessing of 3D CT volumetric scan for L3 localization
    Generates frontal or sagittal maximum intensity projections (MIPs) 
    Performs HU thresholding for bone window to enhance contrast between spine and soft tissue

    Args:
        sitk_image (SimpleITK.SimpleITK.Image): The CT volume loaded as a SimpleITK image 
        target_spacing (float): Target pixel size (default 1mm x 1mm)
        mode ('frontal', 'sagittal'): Plane of projection for MIP 
        threshold (True, False): Whether to perform HU thresholding 

    Returns:
        image (numpy.ndarray): 2D MIP in the plane of choice
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def normalise_zero_one(image, eps=1e-8):
        image = image.astype(np.float32)
        ret = (image - np.min(image))
        ret /= (np.max(image) - np.min(image) + eps)

        return ret

    def reduce_hu_intensity_range(img, minv=100, maxv=1500):
        img = np.clip(img, minv, maxv)
        img = 255 * normalise_zero_one(img)

        return img

    def preprocess_mip_for_slice_detection(image, spacing, target_spacing, threshold = True):
        image = zoom(image, [spacing[2] / target_spacing, spacing[0] / target_spacing])
        if threshold == True:
            image = reduce_hu_intensity_range(image)

        return image
    
    def extract_mip(image, d=10, s=40):
        image_c = image.copy()

        image_c[:, :s, ] = 0
        image_c[:, -s:, ] = 0
        image_c[:, :, :s] = 0
        image_c[:, :, -s:] = 0

        (_, _, Z) = np.meshgrid(range(image.shape[1]), range(image.shape[0]), range(image.shape[2]))
        M = Z * (image_c > 0)
        M = M.sum(axis=2) / (image_c > 0).sum(axis=2)
        M[np.isnan(M)] = 0
        mask = M > 0
        c = int(np.mean(M[mask]))

        image_frontal = np.max(image_c, axis=1)
        image_sagittal = np.max(image_c[:, :, c - d:c + d], axis=2)[::1, :]

        return image_frontal, image_sagittal

    spacing = sitk_image.GetSpacing()
    direction = sitk_image.GetDirection()
    dx = int(direction[0])
    dy = int(direction[4])
    dz = int(direction[8])

    image = sitk.GetArrayFromImage(sitk_image)[::dx, ::dy, ::dz]

    image_frontal, image_sagittal = extract_mip(image)

    if mode == 'sagittal':
        image = image_sagittal
    else:
        image = image_frontal

    return preprocess_mip_for_slice_detection(image, spacing, target_spacing, threshold)