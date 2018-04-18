import numpy as np
from tqdm import tqdm
import pandas as pd

from smooth_tiled_predictions import predict_img_with_smooth_windowing
from image_processing import post_process_image
from image_resize import upsample_masks, resize_image
from image_windows import stitch_all_images
from augment import undo_augment_and_average

def predict_one_window_averaged(predict_model, image, window_size, num_output_channels):
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Input image is expected to be a square, pad width or height as needed,
    # the padding will be cropped before returning the predicted image
    if image_height != image_width:
        pad_height = 0
        pad_width = 0
        if image_height > image_width:
            pad_width = image_height - image_width
        else:
            pad_height = image_width - image_height
        image = np.pad(image, ((0,pad_height),(0,pad_width),(0,0)), 'edge')

    # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small
    # patches with overlap, called once with all those image as a batch outer dimension. Note that
    # model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
    prediction = predict_img_with_smooth_windowing(
        image,
        window_size=window_size,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=num_output_channels,
        pred_func=(lambda img_batch_subdiv: predict_model.predict(img_batch_subdiv)))

    return prediction[0:image_height,0:image_width]

def predict_all_window_averaged(predict_model, images, window_size, num_output_channels):
    """Predict output masks from input images.

    The window-averaged approach is used to generate smooth output predictions. Please
    refer to the following github for a full discussion of this prediction approach:
    https://github.com/Vooban/Smoothly-Blend-Image-Patches

    Args:
        predict_model: The model that should be used to generate the predictions.
        images: All full-sized input images that should be predicted.
        window_size: The window size to use for predictions. This window size must be
          the same size that the predict_model expects for input dimensions. The model
          must accept an input where the height=width=window_size.
        num_output_channels: The number of output channels predicted by the model.

    Returns:
        A list of full-sized predicted masks from the model.
    """
    predict = []
    for image in tqdm(images, total=len(images)):
        new_prediction = predict_one_window_averaged(predict_model, image, window_size, num_output_channels)
        predict.append(new_prediction)

    return predict


def average_model_predictions(images_to_predict, models, weights=[]):
    """ It averages predictions of all given models equally or taking weights
    """
    sum_predictions = 0
    sum_weights = 0
    
    if not weights:
        for model in models:   
            sum_predictions += model.predict(images_to_predict, verbose=1)           
        return sum_predictions / len(models)
    else:
        for index, model in enumerate(models):
            sum_predictions += weights[index]*model.predict(images_to_predict, verbose=1)
            sum_weights += weights[index]
        return sum_predictions / sum_weights

def predict_restore_to_fullsize(predictions, image_sizes):
    """ Restore predicted image windows to full-size. Average results from
    all rotations.
    """
    predictions = undo_augment_and_average(predictions)
    predictions_full_size = stitch_all_images(predictions, image_sizes)

    return predictions_full_size

def convert_prediction_to_labels(prediction, image_process, ori_image_size):
    predict_mask = prediction[:,:,0]
    predict_contour = prediction[:,:,1]
    if image_process.shape[0] != ori_image_size[0]:
        image_process = resize_image(image_process, ori_image_size[0], ori_image_size[1])
        predict_mask = resize_image(predict_mask, ori_image_size[0], ori_image_size[1])
        predict_contour = resize_image(predict_contour, ori_image_size[0], ori_image_size[1])
    
    label = post_process_image(image_process, predict_mask, predict_contour)
    
    return [label]


def add_labels_to_dataframe(cluster_df, image_column='image_process'):
    # Add labels as two steps - the first uses apply since we need to access two columns
    # from the dataframe, and so need to return a list of 2-d images, even though there
    # is only a single entry in each list. The second call (map) converts the 'list' into
    # a plain 2-d numpy array
    cluster_df['label'] = cluster_df.apply(lambda x: convert_prediction_to_labels(x['prediction'], x[image_column], x['size']),axis=1)
    cluster_df['label'] = cluster_df['label'].map(lambda x: x[0])
    return cluster_df