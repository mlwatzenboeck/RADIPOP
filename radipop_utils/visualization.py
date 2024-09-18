import numpy as np 
import matplotlib.pyplot as plt


# Function to apply mask overlays to an axis
def apply_mask_overlay(ax, img_slice, mask_slices, mask_colors):
    """
    Apply mask overlays to an axis with specified colors.

    :param ax: Matplotlib axis object
    :param img_slice: Image slice to display
    :param mask_slices: List of mask slices
    :param mask_colors: List of RGBA colors corresponding to each mask
    """
    ax.imshow(img_slice, cmap='gray')
    ax.axis('off')  # Remove the axis numbering

    for mask_slice, color in zip(mask_slices, mask_colors):
        rgba_image = np.zeros((*img_slice.shape, 4), dtype=np.float32)
        rgba_image[..., 0] = color[0]  # Red channel
        rgba_image[..., 1] = color[1]  # Green channel
        rgba_image[..., 2] = color[2]  # Blue channel
        rgba_image[..., 3] = mask_slice * color[3]  # Alpha channel
        ax.imshow(rgba_image, alpha=color[3])


def overlay_masks(img_slice, mask_slices, mask_colors, facecolor='white'):
    """
    Create a plot with a single image overlayed with masks.

    :param img_slice: Image slice to display
    :param mask_slices: List of mask slices
    :param mask_colors: List of RGBA colors corresponding to each mask
    :return: Matplotlib figure and axis
    """
    fig, ax = plt.subplots(facecolor=facecolor)
    apply_mask_overlay(ax, img_slice, mask_slices, mask_colors)
    # ax.set_facecolor('black')  # Set the background color of the subplot to black
    return fig


def overlay_masks_and_show_bare_image_next_to_it(img_slice, mask_slices, mask_colors, figsize=(12, 6), facecolor='white'):
    """
    Create a plot showing both the bare image and the overlaid masks side by side.

    :param img_slice: Image slice to display
    :param mask_slices: List of mask slices
    :param mask_colors: List of RGBA colors corresponding to each mask
    :return: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor=facecolor)  # Set the background color of the figure to black

    # Show the bare image
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_facecolor(facecolor)  # Set the background color of the subplot to black
    axes[0].axis('off')  # Remove axis numbering

    # Overlay masks
    apply_mask_overlay(axes[1], img_slice, mask_slices, mask_colors)
    axes[1].set_facecolor(facecolor)  # Set the background color of the subplot to black

    # Remove the space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Remove extra padding and margins around the subplots
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Adjust the layout manually for tight fit without gaps
    plt.tight_layout(pad=0)

    return fig



def crop_image(image, top_ratio=0.12, bottom_ratio=0.2, left_ratio=0.05, right_ratio=0.02):
    height, width = image.shape
    top = int(height * top_ratio)
    bottom = int(height * (1 - bottom_ratio))
    left = int(width * left_ratio)
    right = int(width * (1 - right_ratio))
    return image[top:bottom, left:right]