import os
import sys
from glob import glob 
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from image_windows import split_image_to_windows, stitch_single_image, stitch_all_images
from skimage.io import imshow

from sklearn.cluster import KMeans
from skimage.segmentation import random_walker
from scipy.ndimage.morphology import binary_fill_holes

"""
Examples:

How to K-means train images to 2 clusters based on colors
img_df = create_color_features(img_df)
img_df, cluster_maker = create_color_clusters(img_df, 2)

How to perform the same K-means on test images
test_img_df =  create_color_features(test_img_df)
test_img_df, _ = create_color_clusters(test_img_df, 2, cluster_maker)

How to process train/test images, for now
process_images(img_df)
process_images(test_img_df)

How to cluster train/test images to gray/color data frames
cluster_train_df_list = split_cluster_to_group(img_df, 2)
cluster_test_df_list = split_cluster_to_group(test_img_df, 2)

For test images, you should get 53 gray images and 12 color images 
print(len(cluster_test_df_list[0]))
print(len(cluster_test_df_list[1]))

"""

def rgb_clahe(in_rgb_img):
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr[:,:,[2,1,0]]

def rgb_clahe_justl(in_rgb_img):
    bgr = in_rgb_img[:,:,[2,1,0]] # flip r and b
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(lab[:,:,0])
    return img

def invert_image(image):
    if (np.average(image) > 127):
        return cv2.bitwise_not(image)
    return image


def mark_contours(mask):
    """Pass in a 2-dimensional grayscale mask, mark contours of the mask,
    and return the 2-dimensional grayscale contoured mask
    """
    padded = 2
    mask = mask.astype(np.uint8)*255
    
    background = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    padded_background = np.pad(background.copy(), ((padded,padded), (padded,padded)), 'edge')
    background_rgb = cv2.cvtColor(padded_background, cv2.COLOR_GRAY2RGB)

    padded_mask = np.pad(mask.copy(), ((padded,padded), (padded,padded), (0,0)), 'edge')
    
    _,thresh = cv2.threshold(padded_mask,127,255,cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contoured_rgb = cv2.drawContours(background_rgb,contours,-1,(255,255,255),1) 
    contoured_gray = cv2.cvtColor(contoured_rgb, cv2.COLOR_RGB2GRAY)    
    # workaround due to OpenCV issue, the contour starts from the 1st pixel
    contoured_mask = contoured_gray[padded:-padded,padded:-padded]

    return contoured_mask  


def mark_mask_on_image(mask, image, color=(255,0,0)):
    """ Mark red contours of the given masks on the given image
    """
    mask = mask.astype(np.uint8)*255
    image_color = image.copy()
    if image.ndim is 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    _,thresh = cv2.threshold(mask,127,255,cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    return cv2.drawContours(image_color,contours,-1,color,1)
    

def process_images(img_df):
    """It adds a column to the given data frame for storing processed images
    It amplifies regional attributes"""
    img_df['image_process'] = img_df['image'].map(lambda x: invert_image(rgb_clahe_justl(x))[:,:, np.newaxis])
    return img_df
    

def preprocess_image(image):
    """Preprocesses input image to a consistent input format for the model to learn from.

    Preprocessing converts the 3 source image RGB channels down to a single gray-scale
    channel, as well as inverts the background if needed. This ensures that cell the
    background intensity is more consistent, allowing the neural network to learn
    distinguishing features better.

    Args:
        image (numpy 3d matrix): A numpy array containing a single image.
    
    Returns:
        The input image containing only a single channel and possibly with the
        background intensity inverted.
    """
    image = rgb_clahe_justl(image)
    image = invert_image(image)
    image = image[:,:, np.newaxis]
    return image

def renumber_labels(label_img):
    """ Re-number nuclei in a labeled image so the nuclei numbers are unique and consecutive.
    """
    new_label = 0
    for old_label in np.unique(label_img):
        if not old_label == new_label:
            label_img[label_img == old_label] = new_label
        new_label += 1

    return label_img
        

def post_process_image(image, mask, contour):
    """ Watershed on the markers generated on the sure foreground to find all disconnected objects
    The (mask - contour) is the true foreground. We set the contour to be unknown area. 
    Index of contour = -1
    Index of unkown area = 0
    Index of background = 1  -> set back to 0 after watershed
    Index of found objects > 1
    """
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    new_contour = (contour*255).astype(np.uint8)
    new_mask = (mask*255).astype(np.uint8)
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_OPEN, kernel, iterations=1)
  

    _, thresh_mask = cv2.threshold(new_mask,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh_contour = cv2.threshold(new_contour,0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    sure_background = cv2.dilate(thresh_mask,kernel,iterations=3)
    
    sure_foreground = cv2.subtract(thresh_mask, thresh_contour)
    mask_plus_contour = cv2.add(thresh_mask, thresh_contour)
    mask_plus_contour = cv2.cvtColor(mask_plus_contour, cv2.COLOR_GRAY2RGB)

    unknown = cv2.subtract(sure_background, sure_foreground)
    # Marker labelling
    output = cv2.connectedComponentsWithStats(sure_foreground)
    labels = output[1]
    stats = output[2]
    # Add one to all labels so that sure background is not 0, 0 is considered unknown by watershed
    # this way, watershed can distinguish unknown from the background
    labels = labels + 1
    labels[unknown==255] = 0

    try:
        # random walker on thresh_mask leads a lot higher mean IoU but lower LB
        #labels = random_walker(thresh_mask, labels)   
        # random walker on thresh_mask leads lower mean IoU but higher LB
        labels = random_walker(mask_plus_contour, labels, multichannel=True)   

    except:
        labels = cv2.watershed(mask_plus_contour, labels)

    labels[labels==-1] = 0
    labels[labels==1] = 0
    labels = labels -1
    labels[labels==-1] = 0
    # discard nuclei which are too big or too small
    mean = np.mean(stats[1:,cv2.CC_STAT_AREA])

    for i in range(1, labels.max()):
         if stats[i, cv2.CC_STAT_AREA] > mean*10 or stats[i, cv2.CC_STAT_AREA] < mean/10:
            labels[labels==i] = 0
            
    labels = renumber_labels(labels)
        
    return labels


def create_color_features(img_df):
    img_df['Red'] = img_df['image'].map(lambda x: np.mean(x[:,:,0]))
    img_df['Green'] = img_df['image'].map(lambda x: np.mean(x[:,:,1]))
    img_df['Blue'] = img_df['image'].map(lambda x: np.mean(x[:,:,2]))
    img_df['Gray'] = img_df['image'].map(lambda x: np.mean(x[:,:,0:2]))
    img_df['Red-Green'] = img_df['image'].map(lambda x: np.mean(x[:,:,0]-x[:,:,1]))
    img_df['Red-Green-Sd'] = img_df['image'].map(lambda x: np.std(x[:,:,0]-x[:,:,1]))
    return img_df


def create_color_clusters(img_df,  cluster_count = 2, cluster_maker=None, 
                          colors=['Green', 'Red-Green', 'Red-Green-Sd']):
    """ Cluster images based on color features. 
    cluster_count: K  of K-means
    cluster_maker: previous k-means model 
    colors: categories for clustering images, by default it splits images to 
        color and grayscale clusters
    """
    if cluster_maker is None:
        cluster_maker = KMeans(cluster_count, random_state=42)
        cluster_maker.fit(img_df[colors])
        
    img_df['cluster-id'] = np.argmin(cluster_maker.transform(img_df[colors]),-1)
  
    return img_df, cluster_maker


def split_cluster_to_group(img_df, cluster_count, column='cluster-id'):
    """ Pass a data frame and return a list of clustered data frames
    For example, it returns a grayscale img_df and a color img_df 
    """
    cluster_df_list = []
    grouper = img_df.groupby([column])
    for _, cluster_df in grouper:
        cluster_df_list.append(cluster_df)

    assert(len(cluster_df_list) == cluster_count)
    return cluster_df_list




   