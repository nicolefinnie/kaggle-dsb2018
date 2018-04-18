import os
import sys
from glob import glob 
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.io import imshow

from image_processing import mark_contours, preprocess_image

"""
Examples:
How to load train/test images as a data frame
img_df = read_images_as_dataframe(TRAIN_PATH)
test_img_df = read_images_as_dataframe(TEST_PATH)

How to add masks/contours to the train data frame
read_masks_to_dataframe(TRAIN_PATH, img_df)

How to visualize some sampled train/test clustered images
sample_cluster_image(img_df, 3)
plt.show()
sample_cluster_image(test_img_df, 3)
plt.show()

For test images, you should get 53 gray images and 12 color images 
print(len(cluster_test_df_list[0]))
print(len(cluster_test_df_list[1]))


"""


def read_image(base_path, image_name, num_channels, preprocess):
    """Read a single image from disk.

    After the image is read, it will be preprocessed if desired.

    Args:
        base_path: The path that contains the image to be read.
        image_name: The name of the image to read, with no filename extension.
        num_channels: The number of channels from the input image to read.
        preprocess: True to pre-process the image after reading.

    Returns:
        The image, possibly preprocessed, that was read from disk.
    """
    img = cv2.imread(os.path.join(base_path, image_name + '.png'))[:,:,:num_channels]
    if preprocess:
        img = preprocess_image(img)
    return img

def read_single_image_masks(mask_path):
    """Read masks for a single input image.

    A mask will be formed by combining all the separate masks in the provided path.

    Args:
        mask_path: The path that contains the masks to be read.
        mask_height: The image height of the masks.
        mask_width: The image width of the masks.

    Returns:
        The combined full-size mask.
    """
    layered_mask = None
    contoured_mask = None
    for mask_file in next(os.walk(mask_path))[2]:
        next_mask = cv2.imread(os.path.join(mask_path, mask_file), cv2.IMREAD_UNCHANGED)
        next_mask = np.expand_dims(next_mask.astype(dtype=np.bool), axis=-1)
        # Any mask pre-processing should be done here on the full-size masks.
        # At this point, we have the composition of all masks for a single image
        # available in 'layered_mask', and the new single nuclei mask available
        # as 'next_mask'. 'contoured_mask' is just the layered_mask marked with 
        # all contours
        next_contoured_mask = np.expand_dims(mark_contours(next_mask).astype(dtype=np.bool), axis=-1)
       
        if layered_mask is None:
            layered_mask = next_mask
            contoured_mask = next_contoured_mask
        else:
            layered_mask = np.maximum(layered_mask, next_mask)
            contoured_mask = np.maximum(contoured_mask, next_contoured_mask)
            
    return np.append(layered_mask, contoured_mask, axis=-1)

def read_split_channel_image(image_path, image_dtype):
    """Read channel images from disk and combine into a single multi-channel output image.

    Args:
        image_path: The path that contains the image channels to be read.
        image_dtype: The expected data type for the image channels.

    Returns:
        The combined multi-channel image.
    """
    layered_image = None
    (_, _, filenames) = next(os.walk(image_path))   
    if not filenames:
        raise ValueError('Empty directory ' + image_path + ' reading prepared images.')
    for image_file in sorted(filenames):
        next_channel = np.atleast_3d(cv2.imread(os.path.join(image_path, image_file), cv2.IMREAD_UNCHANGED))
        if layered_image is None:
            layered_image = next_channel.astype(dtype=image_dtype)
        else:
            layered_image = np.concatenate((layered_image, next_channel.astype(dtype=image_dtype)), axis=2)

    return layered_image

def read_single_image_prepared_masks(mask_path):
    """Read pre-processed masks for a single input image.

    A mask will be formed by combining all the separate masks in the provided path.
    Unlike read_single_image_masks, this function will store each image as
    a separate channel in the output mask. Files are read in alphabetical order,
    so if there are two files named channel1 and channel2, mask1 will be the first
    channel in the output image, and mask2 will be the second channel.

    Args:
        mask_path: The path that contains the masks to be read.

    Returns:
        The combined full-size mask.
    """
    return read_split_channel_image(mask_path, np.bool)

def read_or_prepare_single_image_masks(path, image_id, save_masks=True, force_mask_regen=False):
    """Read / prepare single mask, also works for data frame
    """
    mask = None
    prep_masks_path = os.path.join(path, 'prep_masks')
    if os.path.isdir(prep_masks_path) and not force_mask_regen:
        mask = read_single_image_prepared_masks(prep_masks_path)
    else:
        mask = read_single_image_masks(os.path.join(path, 'masks'))
        if save_masks:
            write_image(prep_masks_path, image_id, mask, True)

    return mask

def read_masks(base_path, image_ids, sizes, save_masks=True, force_mask_regen=False):
    """Read all mask data for all images.

    All full-size masks will be read from disk. One output mask will be created
    for each input image ID.

    Args:
        base_path: The base path containing the image ID subdirectories, and underneath
          each image ID subdirectory there is expected to be a 'masks' subdirectory.
        image_ids: List of all image IDs.
        sizes: A list containing the image height and width.

    Returns:
        A list of all full-sized image masks.
    """
    print('Reading masks from ' + base_path)
    sys.stdout.flush()
    masks = []
    for id_ in tqdm(image_ids, total=len(image_ids)):
        masks.append(read_or_prepare_single_image_masks(os.path.join(base_path, id_), id_, save_masks, force_mask_regen))

    return (masks)

def read_images(base_path, image_ids, num_channels, preprocess):
    """Read all images from disk.

    All full-size images will be read from disk.

    Args:
        base_path: The base path containing the image ID subdirectories, and underneath
          each image ID subdirectory there is expected to be an 'images' subdirectory.
        image_ids: List of all image IDs.
        num_channels: The number of channels expected in each image file.
        preprocess: True if the input images should also be pre-processed.
        sizes: A list containing the image height and width.

    Returns:
        A pair of lists, the first element containing a list of all full-sized images, and
        the second element containing a list of the image dimensions.
    """
    print('Reading images from ' + base_path)
    sys.stdout.flush()
    images = []
    sizes = []
    for id_ in tqdm(image_ids, total=len(image_ids)):
        img = read_image(os.path.join(base_path, id_, 'images'), id_, num_channels, preprocess)
        images.append(img)
        sizes.append([img.shape[0], img.shape[1]])
        
    return (images, sizes)

def read_prediction_image(prediction_path):
    """Read predicted image from disk.

    Args:
        prediction_path: The path that contains the prediction channels to be read.

    Returns:
        The combined prediction image.
    """
    return (read_split_channel_image(prediction_path, np.uint16).astype(np.float32) / 65535)

def write_image(base_path, image_name, image, separate_channels):
    """Write a single image to disk.

    Args:
        base_path: The path that the image should be written to. This path will
          be created if it does not exist.
        image_name: The name of the image to write, with no filename extension.
        image: The image to write.
    """
    if not os.path.isdir(base_path):
        os.makedirs(base_path)

    write_image = image
    if write_image.dtype == np.bool:
        write_image = write_image.astype(np.uint8)*255

    if separate_channels:
        for n in range(image.shape[2]):
            cv2.imwrite(os.path.join(base_path, 'channel_' + str(n) + '.png'), write_image[:,:,n])
    else:
        cv2.imwrite(os.path.join(base_path, image_name + '.png'), write_image)

def write_images(base_path, image_ids, subpath, images, separate_channels):
    """Writes all images to disk.

    Args:
        base_path: The base path containing the image ID subdirectories, and underneath
          each image ID subdirectory there is expected to be a 'subpath' subdirectory.
          The new files will be written to this 'subpath' subdirectory. This 'subpath'
          directory will be created if it does not exist.
        image_ids: List of all image IDs.
        subpath: Subdirectory under 'base_path + id_', i.e. 'masks' or 'images'
        images: The list of all images to write.
        separate_channels: True if each image should be written to a separate channel,
          in which case they will be named 'channel_0' .. 'channel_n-1', where 'n' is the
          number of channels detected in the image.
    """
    print('Writing images to ' + base_path)
    sys.stdout.flush()
    for id_, img in tqdm(zip(image_ids, images), total=len(image_ids)):
        write_image(os.path.join(base_path, id_, subpath), id_, img, separate_channels)

def save_prediction_images(df, base_path):
    """ Save all predicted images to disk.

    Args:
        df: Data frame that contains the predicted images.
        base_path: Base path where predictions should be saved. Images will be
          saved in a subdirectory under this base path, and the subdirectory
          name is the imageID.
    """
    for img_id, prediction in zip(df.imageID, df.prediction):
        prediction_scaled = (prediction * 65535).astype(np.uint16)
        write_image(os.path.join(base_path, str(img_id)), img_id, prediction_scaled, True)

def load_prediction_images_to_df(df, base_path):
    """ Load the predicted images saved on disk into the given data frame.

    Args:
        df: The data frame in which the images should be stored
        base_path: The base path in which the predicted images are stored.

    Returns:
        A data frame with the 'prediction' column which contains the
        predicted images read from disk.
    """
    df['prediction'] = df['imageID'].map(lambda x: read_prediction_image(os.path.join(base_path, x)))
    return df

def read_images_as_dataframe(path):
    """ Read all training or test images as Pandas Dataframe
    
    path - data path
    Training: True for training data, it will build mask columns as labelled data 
    save_masks: True - generated multi-channel mask/contour files will be saved on the disk
    force_mask_regen: False - if masks/contour files exist, they won't re-generate again
    """
    
    images = glob(os.path.join(path, '*', 'images', '*.png'))
    img_df = pd.DataFrame({'path':images})
    img_id = lambda in_path: in_path.split(os.sep)[-3]
    img_df['imageID'] = img_df['path'].map(img_id)
    img_df['image'] = img_df['path'].map(lambda x: cv2.imread(x)[:,:,:3])
    img_df['size'] = img_df['image'].map(lambda x: (x.shape[0],x.shape[1])) 
   
    return img_df

def read_masks_to_dataframe(path, img_df, save_masks=True, force_mask_regen=False):
    """ Read all masks and contoured masks to the given data frame
    !!! it changes the data frame in place !!!
    save_masks: True - generated multi-channel mask/contour files will be saved on the disk
    force_mask_regen: False - if masks/contour files exist, they won't re-generate again
    """ 
    img_df['mask_train'] = img_df['imageID'].map(lambda x: read_or_prepare_single_image_masks(os.path.join(path, x), x, save_masks, force_mask_regen))
    img_df['mask'] = img_df['mask_train'].map(lambda x: x[:,:,0])
    img_df['contour'] = img_df['mask_train'].map(lambda x: x[:,:,1])

def show_image(image):
    """Show a graphical representation of the input image.

    The input image can be a multi-channel image, a single-channel image, or
    a 2-d numpy matrix.

    Args:
        image: The image to be displayed.
    """
    if len(image.shape) == 2:
        imshow(image)
    elif image.shape[2] == 1:
        imshow(np.squeeze(image))
    elif image.shape[2] == 2:
        # Images with 2 channels are not valid images, but each channel may
        # have interesting data. Show each channel separately
        imshow(np.squeeze(image[:,:,0]))
        plt.show()
        imshow(np.squeeze(image[:,:,1]))
    else:
        imshow(image)
    plt.show()

def sample_cluster_image(img_df, image_per_column):
    """ Show some sampled clustered images"""
    grouper = img_df.groupby(['cluster-id'])
    _, m_axs = plt.subplots(image_per_column, len(grouper), figsize = (20, 10))
    for (c_group, clus_group), c_ims in zip(grouper, m_axs.T):
        c_ims[0].set_title('Group: {}\n'.format(c_group))
        for (_, clus_row), c_im in zip(clus_group.sample(image_per_column, replace = True).iterrows(), c_ims):
            c_im.imshow(clus_row['image'])
            c_im.axis('off')

def return_images_without_outliers(img_df):
    """ Filter out training images that are known to have bad/invalid masks. """
    outliers = ['7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80',
                'adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df',
                '12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40']

    new_df = img_df.copy()
    for id in range(len(outliers)):
        new_df.drop(new_df[new_df['imageID'] == outliers[id]].index, inplace=True)

    return new_df

def split_train_val_set(img_df, val_split=0.03):
    np.random.seed(42)
    imageIDs = np.sort(img_df['imageID'].values)
    val_set_count = int(len(img_df.index)*val_split)

    val_set_imageIDs = np.random.choice(imageIDs, val_set_count)
    val_set = img_df[img_df['imageID'].isin(val_set_imageIDs)]

    return img_df.drop(val_set.index), val_set