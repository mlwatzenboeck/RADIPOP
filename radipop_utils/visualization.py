import numpy as np 
import matplotlib.pyplot as plt


def overlay_masks(img_slice, mask_slices, mask_colors):
    fig, ax = plt.subplots()
    ax.imshow(img_slice, cmap='gray')
    ax.axis('off')  # Remove the axis numbering

    # Overlay each mask with the corresponding color and transparency
    for mask_slice, color in zip(mask_slices, mask_colors):
        rgba_image = np.zeros((*img_slice.shape, 4), dtype=np.float32)
        rgba_image[..., 0] = color[0]  # Red channel
        rgba_image[..., 1] = color[1]  # Green channel
        rgba_image[..., 2] = color[2]  # Blue channel
        rgba_image[..., 3] = mask_slice * color[3]  # Alpha channel
        
        # Overlay the mask slice with transparency
        ax.imshow(rgba_image, alpha=color[3])
    
    return fig




def crop_image(image, top_ratio=0.12, bottom_ratio=0.2, left_ratio=0.05, right_ratio=0.02):
    height, width = image.shape
    top = int(height * top_ratio)
    bottom = int(height * (1 - bottom_ratio))
    left = int(width * left_ratio)
    right = int(width * (1 - right_ratio))
    return image[top:bottom, left:right]