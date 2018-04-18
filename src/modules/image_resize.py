import sys
import numpy as np

import cv2
from tqdm import tqdm
from image_windows import split_image_to_windows, stitch_all_images
from image_io import show_image

def get_mean_cell_size(mask_contours):
    nuclei_sizes = []
    for mask_contour in mask_contours: 
        mask = mask_contour[:,:,0]
        contour = mask_contour[:,:,1]
        new_mask = (mask*255).astype(np.uint8)
        new_contour = (contour*255).astype(np.uint8)
        true_foreground = cv2.subtract(new_mask, new_contour)
        output = cv2.connectedComponentsWithStats(true_foreground)
        nuclei_sizes.append(np.mean(output[2][1:,cv2.CC_STAT_AREA]))
    return nuclei_sizes


def resize_image(image, new_image_height, new_image_width):
    """Resize an input image into the provided dimensions.

    Resizing is either implemented by splitting an image into multiple windows,
    or by down-scaling/up-scaling the image. The window approach preserves the
    original image dimensions, however returns multiple windows as output.
    Down-scaling/up-scaling generates only one output image, however the scaling
    may produce undesirable image artifacts.

    Args:
        image: The full-size input image.
        new_image_height: The desired height of the output image(s).
        new_image_width: The desired width of the output image(s).
  
    Returns:
        Numpy 4-d array, dimensions are the window index, window height, window
        width, channel data. The returned shape[0] describes how many total
        windows were created.
    """
    
    # open cv cannot handle the bool data type used by masks. Convert to uint8 for the
    # resize operation if needed. Note also that opencv expects dimensions in
    # width x height format (not height x width, which numpy uses!)
    
    # shrink, use area
    interpolation = cv2.INTER_AREA
    if image.shape[0] < new_image_height:
        # enlarge, use cubic
        interpolation = cv2.INTER_CUBIC

    if image.dtype == np.bool:
        resized_img = cv2.resize(image.astype(np.uint8),
                                    (new_image_width, new_image_height),
                                    interpolation=interpolation).astype(np.bool)
    else:
        resized_img = cv2.resize(image, (new_image_width, new_image_height), interpolation=interpolation)
    return np.atleast_3d(resized_img)

def resize_images(images, new_size):
    resized_images = [resize_image(image, new_size, new_size) for image in images]
    return resized_images


def window_images(images, new_height, new_width):
    """Resize input images into the provided dimensions.

    Resizing is either implemented by splitting images into multiple windows,
    or by down-scaling/up-scaling the images. The window approach preserves the
    original image dimensions, however returns multiple windows for each image.
    Down-scaling/up-scaling generates only one output image, however the scaling
    may produce undesirable image artifacts.

    Args:
        images: The full-size input image.
        new_height: The desired height of the output image.
        new_width: The desired width of the output image.
        split_to_windows: True to generate output windows, False to scale.

    Returns:
        Numpy 4-d array, dimensions are the window/image index, window height, window
        width, channel data. The returned shape[0] describes how many total
        windows/images were created.
    """

    resized_images = []
    for img in images:
        resized_images.extend(split_image_to_windows(img, new_height, new_width))
                
    return np.asarray(resized_images, dtype=np.uint8)


def upsample_masks(masks, sizes):
    """Upscales the provided masks to the provided sizes.

    Args:
        masks: List of scaled input images/masks, expected to contain only
          a single channel.
        sizes: List of dimensions that each input image/mask should be
          scaled to.

    Returns:
        A list of numpy 2-d matrices, where each entry in the list corresponds
        to the full-sized image/mask after being scaled.
    """
    upsampled = []
    for i in range(len(masks)):
        # Note - opencv expects dimensions in width * height format (not h x w!)
        upsampled.append(cv2.resize(np.squeeze(masks[i]), 
                                    (sizes[i][1], sizes[i][0]),
                                    interpolation = cv2.INTER_LINEAR))
    return upsampled

def restore_image_to_fullsize(images, sizes, use_image_windows):
    """Restores images back to their original full size.

    Args:
        images: The images that should be restored to their original sizes.
        sizes: List of original image dimensions.
        use_image_windows: True if the input list contains windows, and the
          resulting image should be formed by stitching the windows back
          together. False if each image is a scaled version of the original
          image, and should be restored by up-scaling/down-scaling.

    Returns:
        A list of numpy matrices, where each entry in the list corresponds
        to the full-sized image after being restored to the original size.
    """
    if use_image_windows:
        fullsize_images = stitch_all_images(images, sizes)
    else:
        fullsize_images = upsample_masks(images, sizes)
    return fullsize_images

