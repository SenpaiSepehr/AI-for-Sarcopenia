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

def slice_selection(z, spacing, height, offset=0):
    """Calculates the slice index corresponding to a longitudinal coordinate 
    Args:
        z (float): The predicted L3 coordinate on the longitudinal axis
        spacing (float): Slice thickness of the original CT volume 
        height (float): Height of the original CT volume
        offset (float): Offset of the slices in the original CT volume
    
    Returns: Index of the slice in the volume corresponding to z

    """

    return int((height-z) // spacing[2] + offset)
    
def show_slice_window(slice, level, window):
    """ Performs HU thresholding on an CT image slice 
    Args:
        slice (numpy.darray): 2D CT slice
        level (float): Center of the HU window 
        window (float): Width of the HU window
        
    Returns: 
        slice (numpy.darray): The thresholded slice
    """
    max = level + window/2
    min = level - window/2
    slice = slice.clip(min,max)
    return slice
