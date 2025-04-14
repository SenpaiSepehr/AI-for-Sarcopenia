import os
import numpy as np
import torch
import SimpleITK as sitk
import cv2
from model import TResUnet

# segmentation weights
weight_segment = "/path/to/your/segmentation_weights.pth"
ct_path = "/path/to/your/volume.nii.gz"

# Load your model
def load_model():
    model = TResUnet(patch_size=4)
    model.load_state_dict(torch.load(weight_segment, map_location=torch.device('cpu')))
    model.eval()
    return model

segment_model = load_model()

# Run inference on the L3 slice
def run_inference(image):
    # Image size
    size = (256, 256)
    
    # Preprocess image
    image = cv2.resize(image, size)
    save_img = image.copy()
    
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to the format expected by your model
    image = np.transpose(image, (2, 0, 1))
    image = image/255.0
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    
    with torch.no_grad():
        # Run inference with heatmap
        heatmap, y_pred = segment_model(image, heatmap=True)
        y_pred = torch.sigmoid(y_pred)
        
        # Convert prediction to binary mask
        y_pred = y_pred[0].cpu().numpy()
        y_pred = np.squeeze(y_pred, axis=0)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.uint8) * 255
        
                    # Binary mask (1 where segmented, 0 elsewhere)
        binary_mask = y_pred > 0

        # Count number of segmented pixels
        pixel_count = np.sum(binary_mask)
        sitk_image = sitk.ReadImage(ct_path)
        spacing = sitk_image.GetSpacing()  
        x_spacing, y_spacing = spacing[0], spacing[1]
        mm2_per_pixel = x_spacing * y_spacing
        # Convert to mm² (1 pixel = 1 mm²)
        area_mm2 = pixel_count * mm2_per_pixel  # this is redundant but clear

        # Optionally: convert to cm² or m²
        area_cm2 = area_mm2 / 100

        
        # Convert grayscale mask to RGB for display
        mask_rgb = np.zeros((y_pred.shape[0], y_pred.shape[1], 3), dtype=np.uint8)
        mask_rgb[:,:,1] = y_pred  # Green channel for visibility
        
        # Prepare heatmap for visualization
        # Check if heatmap is a tensor or numpy array
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap[0].cpu().numpy()
        else:
            # If already numpy, just take the first item if it's batched
            if heatmap.ndim > 3:
                heatmap = heatmap[0]
                
        heatmap = cv2.resize(heatmap, size)
        # Normalize heatmap to 0-255 range
        heatmap = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8) * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Create overlay visualization
        if len(save_img.shape) == 2:
            save_img_rgb = cv2.cvtColor(save_img, cv2.COLOR_GRAY2RGB)
        else:
            save_img_rgb = save_img
            
        # Create a visualization with all results
        alpha = 0.5
        overlay = cv2.addWeighted(save_img_rgb, 1, mask_rgb, alpha, 0)
        
        
        return overlay, mask_rgb, heatmap_colored, area_cm2