import numpy as np
from image_resize import resize_image
from imgaug import augmenters as iaa
from skimage import util as skiutil
import image_processing as impr
import cv2

def augment_max(images):
    """Augment the given training data as much as possible.

    All available augmentation mechanisms will be used to generate as much training
    data as possible.

    Args:
        images (numpy array): A numpy array containing the input images.
            Dimensions are expected to be the number of samples, the height,
            the width, and the number of channels (i.e. 3 for RGB images, 1 for
            grayscale images).
    
    Returns:
        The augmented images.
    
    """
    all_images = []
    images_1 = np.rot90(images, axes=(1, 2), k=1)
    images_2 = np.rot90(images, axes=(1, 2), k=2)
    images_3 = np.rot90(images, axes=(1, 2), k=3)
    images_4 = images[:, :, ::-1]
    images_5 = np.rot90(images_4, axes=(1, 2), k=1)
    images_6 = np.rot90(images_4, axes=(1, 2), k=2)
    images_7 = np.rot90(images_4, axes=(1, 2), k=3)

    all_images.extend(images)
    all_images.extend(images_1)
    all_images.extend(images_2)
    all_images.extend(images_3)
    all_images.extend(images_4)
    all_images.extend(images_5)
    all_images.extend(images_6)
    all_images.extend(images_7)

    return np.asarray(all_images, dtype=images.dtype)

def undo_augment_and_average(images):
    """Reverses the rotations performed by augment_max.
    
    All images will be averaged together to generate a single image in the
    original orientation.

    Args:
        images (numpy array): A numpy array containing the input images that
            have been augmented from augment_max. Dimensions are expected to be
            the number of samples, the height, the width, and the number of
            channels (i.e. 3 for RGB images, 1 for grayscale images).
    
    Returns:
        Numpy array with the original (un-augmented) number of images, where
        each image is in the original orientation and the intensity is the
        average of each of the corresponding augmented images.
    """
    num_orig = int(images.shape[0] / 8)
    images_1 = np.rot90(images[num_orig:num_orig*2], axes=(1, 2), k=3)
    images_2 = np.rot90(images[num_orig*2:num_orig*3], axes=(1, 2), k=2)
    images_3 = np.rot90(images[num_orig*3:num_orig*4], axes=(1, 2), k=1)
    images_4 = images[num_orig*4:num_orig*5, :, ::-1]
    images_5 = np.rot90(images[num_orig*5:num_orig*6], axes=(1, 2), k=3)[:, :, ::-1]
    images_6 = np.rot90(images[num_orig*6:num_orig*7], axes=(1, 2), k=2)[:, :, ::-1]
    images_7 = np.rot90(images[num_orig*7:], axes=(1, 2), k=1)[:, :, ::-1]

    all_images = np.zeros((num_orig, images.shape[1], images.shape[2], images.shape[3]), dtype=images.dtype)
    for i in range(num_orig):
        mean1 = np.mean(np.array([images[i], images_1[i]]), axis=0)
        mean2 = np.mean(np.array([images_2[i], images_3[i]]), axis=0)
        mean3 = np.mean(np.array([images_4[i], images_5[i]]), axis=0)
        mean4 = np.mean(np.array([images_6[i], images_7[i]]), axis=0)
        mean5 = np.mean(np.array([mean1, mean2]), axis=0)
        mean6 = np.mean(np.array([mean3, mean4]), axis=0)
        mean7 = np.mean(np.array([mean5, mean6]), axis=0)
        all_images[i] = mean7

    return (all_images)

def scale_images_on_nuclei_size(images, nuclei_sizes, scale, min_size, max_size):
    median_size = np.median(nuclei_sizes)
    subsampled_images = []
    for index, image in enumerate(images):
        if nuclei_sizes[index] < median_size:
            subsampled_images.extend(image_pyramid(image, 1/scale, max_size))
        else:
            subsampled_images.extend(image_pyramid(image, scale, min_size))

    return subsampled_images

def image_pyramid(image, scale, border_size):
    pyramid = []
    
    while True:
        height = int(image.shape[0] / scale)
        width = int(image.shape[1] / scale)
        image = resize_image(image, height, width)
        
        # if the resized image does not meet the supplied minimum or maximum
        # size, then stop constructing the pyramid
        if scale > 1.0:
            if height < border_size or width < border_size:
                # we can downsample smaller than the window size
                break
        else: 
            if height > border_size or width > border_size:
                # we scale up at least once even when the upsampled image is bigger than the maxsize 
                pyramid.append(image)   
                break
        pyramid.append(image)      

    return pyramid          


def additive_Gaussian_noise(images, scale):
    transformer = iaa.AdditiveGaussianNoise(scale=(0,scale*255), deterministic=True)
    return augment_on_df(images, transformer)

def speckle_noise(images):
    new_images = []
    for image in images:
        new_images.append( skiutil.random_noise(image, mode='speckle', seed=42) )

    return new_images

def blur(images, sigma):
    transformer = iaa.GaussianBlur(sigma, deterministic=True)
    return augment_on_df(images, transformer)

def get_perspective_transform_sequence(sigma):
    return iaa.Sequential([iaa.PerspectiveTransform(scale=(float(sigma/10), sigma), deterministic=True)], deterministic=True)

def perspective_transform(images, sequence):
    if len(images[0].shape) > 2 and images[0].shape[2] == 2:
        images = [img.astype(dtype=np.uint8) for img in images]
        images = sequence.augment_images(images)
        images = [img.astype(dtype=np.bool) for img in images]
    else:
        images = sequence.augment_images(images)
    return images
    
def greyscale(images, alpha):
    transformer = iaa.Grayscale(alpha=(0.0, alpha), deterministic=True)
    images = [impr.invert_image(img) for img in images]
    return keep_L_channel(augment_on_df(images, transformer))   

def invert(images):
    for index, image in enumerate(images):
        images[index] = cv2.bitwise_not(image)[:,:, np.newaxis]
    return images
    
    
# we expect N x M x 1 dimensional images, so we keep the L channel only
def keep_L_channel(images):
    for index, image in enumerate(images):
        if image.shape[2] == 3:
            bgr = image[:,:,[2,1,0]] # flip r and b
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            images[index] = lab[:,:,0]
            images[index] = images[index][:,:, np.newaxis]
    return images

def augment_on_df(images, transformer):
    new_images = []
    for image in images:
        if len(image.shape) > 2 and image.shape[2] == 2:
            # for two channels mask data
            # openCV doesn't support boolean type
            channel0 = np.expand_dims(image[:,:,0].astype(dtype=np.uint8), axis=-1)
            channel1 = np.expand_dims(image[:,:,1].astype(dtype=np.uint8), axis=-1)
            image = np.append(channel0, channel1, axis=-1)

            image = transformer.augment_images([image])
            image = image[0].astype(dtype=np.bool) 
        else:
            image = transformer.augment_image(image)
        new_images.append(image)

    return new_images


