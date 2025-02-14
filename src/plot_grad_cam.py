import cv2
import numpy as np
from PIL import Image

def plot_grad_cam(img, heatmap):
    # Resize heatmap to match the image dimensions
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB (jet colormap)
    heatmap = np.uint8(255 * heatmap)  # Convert heatmap to range [0, 255]
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply a colormap to the heatmap
    heatmap = np.float32(heatmap) / 255  # Normalize to [0, 1] for overlay
    
    # Overlay the heatmap on the image
    superimposed_img = heatmap + np.float32(img)  # img should be in range [0, 1]
    superimposed_img = superimposed_img / np.max(superimposed_img)  # Normalize
    
    # Convert to PIL Image for Streamlit compatibility
    pil_img = Image.fromarray((superimposed_img * 255).astype(np.uint8))
    
    return pil_img