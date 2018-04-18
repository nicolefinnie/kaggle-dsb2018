# based on: https://www.kaggle.com/wcukierski/example-metric-implementation

import os
from glob import glob
import numpy as np
import pandas as pd
import cv2

from skimage.morphology import label


def create_labeled_masks(path):
    """
    Create labeled masks for calculation of the mean IoU between the ground truth and the prediction.

    :param path: the location to the images and their associated masks

    :return: the ground truth
    """
    ids = next(os.walk(path))[1]
    Y_true = []

    for id in ids:
        file = path + '/{}/images/{}.png'.format(id,id)
        masks = path + '/{}/masks/*.png'.format(id)
        masknames = glob(os.path.join(path, str(id), 'masks') + '/*.png')

        image = cv2.imread(file)
        masks = np.zeros((len(masknames), image.shape[0], image.shape[1]), dtype=np.uint8)
        for ix, maskname in enumerate(masknames):
            masks[ix] = cv2.imread(maskname, cv2.IMREAD_UNCHANGED)
        height, width, _ = image.shape
        num_masks = masks.shape[0]
        # Make a ground truth label image (pixel value is index of object label)
        labels = np.zeros((height, width), np.uint16)
        for index in range(0, num_masks):
            labels[masks[index] > 0] = index + 1

        Y_true.append(labels)

    return Y_true


def mean_IoU(Y_true, Y_pred):
    """
    Calculate the mean IoU score between two lists of labeled masks.
    :param Y_true: a list of labeled masks (numpy arrays) - the ground truth
    :param Y_pred: a list labeled predicted masks (numpy arrays) for images with the original dimensions
    :return: mean IoU score for corresponding images
    """
    image_precs = []
    for y_true,y_pred in zip(Y_true,Y_pred):
        true_objects = len(np.unique(y_true))
        pred_objects = len(np.unique(y_pred))

        # Compute intersection between all objects
        intersection = np.histogram2d(y_true.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(y_true, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union

        # Precision helper function
        def precision_at(threshold, iou):
            matches = iou > threshold
            true_positives = np.sum(matches, axis=1) == 1   # Correct objects
            false_positives = np.sum(matches, axis=0) == 0  # Missed objects
            false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
            tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
            return tp, fp, fn

        # Loop over IoU thresholds
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            tp, fp, fn = precision_at(t, iou)
            p = tp / (tp + fp + fn)
            prec.append(p)

        image_precs.append(prec)
    return [np.mean(image_precs), image_precs]


def add_metrics_to_df(img_df, Y_true, use_label=False, threshold=0.5):
    """
    Calculate the precision at different intersection over union (IoU) thresholds and the mean average value of it
    for each image in the given pandas.DataFrame.

    :param img_df: pandas.DataFrame with the column "prediction" that already contains the labeled 0/1 predictions
    for each pixel in the first channel
    :param Y_true: the ground truth with the original image size (calculated by the function create_labeled_masks)
    :param use_label: True if post-processed 'label' predictions should be used, False to use raw 'prediction'.
    :return: pandas.DataFrame with added metrics for further analysis
    """
    idx = img_df.index
    Y_true_sel = np.take(Y_true, idx)
    if use_label:
        Y_pred = img_df['label'].values
    else:
        Y_pred = img_df['prediction'].values
        Y_pred = [ x[:, :, 0] for x in Y_pred]
        Y_pred =  [label(pred>threshold) for pred in Y_pred]
    precs = mean_IoU(Y_true_sel, Y_pred)[1]
    img_df['IoU_t'] = precs
    img_df['mean_IoU'] = img_df['IoU_t'].apply(lambda x: np.mean(x))

    # Note that the indices in the input data frame are not necessarily contiguous, so use lambda
    # functions to extract/generate one new column at a time.
    for tix, t in enumerate(np.arange(0.5, 1.0, 0.05)):
        img_df['IoU_' + str(t)] = img_df['IoU_t'].apply(lambda x: x[tix])

    return img_df


def write_metrics_to_file(img_df, path):
    """
    Write the calculated metrics to file.

    :param img_df: pandas.DataFrame with added metrics
    :param path: Location to which the metrics should be written.
    """
    columns = []
    for t in np.arange(0.5, 1.0, 0.05):
        columns.append('IoU_' + str(t))

    header = ['imageID', 'IoU_t', 'mean_IoU']
    header.extend(columns)
    img_df.to_csv(path, sep=',', index=True, columns=header)

