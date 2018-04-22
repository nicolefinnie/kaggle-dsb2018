# RLE Source: https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
import os
import numpy as np
import pandas as pd
from skimage.morphology import label

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

def label_to_rles(lab_img):
    # For each nuclei, skipping the first element(background '0' value)
    for i in np.unique(lab_img)[1:]:
        yield rle_encoding(lab_img == i)

def generate_submission(test_ids, predictions, threshold, submission_file):
    """Generates a submission RLE CSV file based on the input predicted masks.

    Args:
        test_ids: List of all test_ids that were predicted.
        predictions: List of all full-size predicted masks, where each pixel is
          represented as a probability between 0 and 1.
        threshold: The threshold above which a predicted probability is considered 'True'.
        submission_file: The output submission file name.
    """
    new_test_ids = []
    rles = []

    for n, id_ in enumerate(test_ids):
        rle = list(prob_to_rles(predictions[n], threshold))
        rles.extend(rle)
        new_test_ids.extend([id_] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(submission_file, index=False)

def generate_submission_from_df_prediction(dfs, threshold, submission_file):
    """Generates a submission RLE CSV file based on the input data frame's predicted image.

    Args:
        dfs: List of all data frames that contain labeled masks to submit.
        threshold: The threshold above which a predicted probability is considered 'True'.
        submission_file: The output submission file name.
    """
    basename, _ = os.path.split(submission_file)
    if not os.path.isdir(basename) and basename:
        os.makedirs(basename) 

    new_test_ids = []
    rles = []

    for df in dfs:
        for img_id, prediction in zip(df.imageID, df.prediction):
            mask = prediction[:,:,0]
            rle = list(prob_to_rles(mask, threshold))
            rles.extend(rle)
            new_test_ids.extend([img_id] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(submission_file, index=False)

def generate_submission_from_df(dfs, submission_file):
    """Generates a submission RLE CSV file based on the labeled mask in a data frame.

    Args:
        dfs: List of all data frames that contain labeled masks to submit.
        submission_file: The output submission file name.
    """
    basename, _ = os.path.split(submission_file)
    if not os.path.isdir(basename) and basename:
        os.makedirs(basename)

    new_test_ids = []
    rles = []

    for df in dfs:
        for img_id, label in zip(df.imageID, df.label):
            if len(np.unique(label)) == 1:
                rles.extend([''])
                new_test_ids.extend([img_id])
            else:
                rle = list(label_to_rles(label))
                rles.extend(rle)
                new_test_ids.extend([img_id] * len(rle))

    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(submission_file, index=False)