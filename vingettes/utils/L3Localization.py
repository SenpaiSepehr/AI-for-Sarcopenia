from ultralytics import YOLO
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
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
import SimpleITK as sitk

def get_L3_prediction(results, test_img, mode = "frontal"): 

    """ Calculates the predicted L3 level from YOLO model results 
    Visualizes the bounding boxes and the L3 longitudinal coordinate on the original MIP used for prediction

    Args:
        results (list): Results from the YOLO prediction on the MIP  
        test_img (string): Path to the original MIP used for predition 
        mode ('frontal', 'sagittal'): Plane of projection for MIP 

    Returns:
        pred (float): Predicted L3 coordinate on the longitudinal axis
    """
        
    r = results[0]
    if mode == "frontal":
        m = "Frontal"
        sum = 0
        n = 0
        for box in r.boxes:
            cls = int(box.cls.tolist()[0])
            xywh = box.xywh.tolist()[0]
            w = xywh[2]
            h = xywh[3]
            y = xywh[1]+(h/2)
            if n < 2:
                if cls == 0:
                    y = xywh[1] + (h/2)
                    sum = sum + xywh[1] + (h/2)
                    n = n + 1
                elif cls == 1:
                    y = xywh[1] - (h/2)
                    sum = sum + (xywh[1] - (h/2))
                    n = n + 1

        pred = sum/n

        im = Image.open(test_img)
        np_im = np.array(im) 
        maxY, maxX, c = np_im.shape

        fig, ax = plt.subplots()
        plt.imshow(im)


        n = 0
        for box in r.boxes:
            cls = int(box.cls.tolist()[0])
            #print(cls)
            xywh = box.xywh.tolist()[0]
            w = xywh[2]
            h = xywh[3]
            x = xywh[0]-(w/2)
            y = xywh[1]-(h/2)

            if cls == 0:
                col = '#E66100'
                n += 1
            elif cls == 1:
                col = '#5D3A9B'
                n += 1

            else:
                col = '#0072ed'

            if n < 3: 
                rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=col, facecolor='none')
            ax.add_patch(rect)

        plt.hlines(y = pred, xmin = 0, xmax = maxX, color = "w", linestyles=(0, (5, 5)), linewidth=2.5)
        plt.ylabel("Height (mm)")
        print(f"{m} prediction: {round(pred,2)} mm")
    
    elif mode == "sagittal":
        m = "Sagittal"
        def find_fourth_largest(numbers):
            for i in range(3):
                largest = max(numbers)
                numbers.remove(largest)
            return np.max(numbers)

        ys = []
        for box in r.boxes:
            xywh = box.xywh.tolist()[0]
            ys.append(xywh[1])

        pred = find_fourth_largest(ys)

        im = Image.open(test_img)
        np_im = np.array(im) 
        maxY, maxX, c = np_im.shape
        fig, ax = plt.subplots()
        ax.imshow(im)

        for box in r.boxes:
            cls = int(box.cls.tolist()[0])
            #print(cls)
            xywh = box.xywh.tolist()[0]
            w = xywh[2]
            h = xywh[3]
            x = xywh[0]-(w/2)
            y = xywh[1]-(h/2)
            if cls == 8:
                col = '#d73e41' #red
            else:
                col = '#f3b400'
            rect = patches.Rectangle((x, y), w, h, linewidth=1.5, edgecolor=col, facecolor='none')
            ax.add_patch(rect)
        
        plt.hlines(y = pred, xmin = 0, xmax = maxX, color = "w", linestyles=(0, (5, 5)), linewidth=2.5)
        plt.ylabel("Height (mm)")
        print(f"{m} prediction: {round(pred,2)} mm")
    
    return pred