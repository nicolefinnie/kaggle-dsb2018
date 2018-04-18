import os
import datetime
import numpy as np
from tqdm import tqdm

from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.utils import multi_gpu_model
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping


smooth = 1.
def dice_coef(y_true, y_pred):
    """Generate the 'Dice' coefficient for the provided prediction.

    Args:
        y_true: The expected/desired output mask.
        y_pred: The actual/predicted mask.

    Returns:
        The Dice coefficient between the expected and actual outputs. Values
        closer to 1 are considered 'better'.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    """Model loss function using the 'Dice' coefficient.

    Args:
        y_true: The expected/desired output mask.
        y_pred: The actual/predicted mask.

    Returns:
        The corresponding loss, related to the dice coefficient between the expected
        and actual outputs. Values closer to 0 are considered 'better'.
    """
    return -dice_coef(y_true, y_pred)

def build_model(input_height, input_width, input_channels, output_channels, min_dropout=0.1):
    """Build a simple U-net Keras model.

    Args:
        input_height: The height of all input images.
        input_width: The width of all input images.
        input_channels: The number of input channels.
        output_channels: The number of predicted/output channels.

    Returns:
        A Keras U-net model that can be used for training.
    """
    inputs = Input((input_height, input_width, input_channels))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(min_dropout) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(min_dropout) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(min_dropout+0.1) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(min_dropout+0.1) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(min_dropout+0.2) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(min_dropout+0.1) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(min_dropout+0.1) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(min_dropout) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(min_dropout) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(output_channels, (1, 1), activation='sigmoid') (c9) # try softmax  with categorial_crossentropy as loss

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_typed_model(model, gpu=1):
    """ Return a multi GPU model if there are more than one GPU
    otherwise return a normal model
    """
    if gpu > 1:
        gpu_model = multi_gpu_model(model, gpu)
        gpu_model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [dice_coef])
        typed_model = gpu_model
    else:
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics = [dice_coef])
        typed_model = model
    return typed_model
    
def save_model(model, model_name_prefix, save_model_path=os.path.curdir):
    if not os.path.isdir(save_model_path):
        os.makedirs(save_model_path)
    model_name = model_name_prefix + '-' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))+'.h5'
    model.save(os.path.join(save_model_path, model_name))
    print("Model saved as " + model_name)
    return model_name


def train_model(X_train, Y_train, X_val, Y_val, epochs=20, gpu=1, earlystopping=5, load_model_path=None, save_model_path=None, min_dropout=0.1):
    """ Build or load a single model and train data on the single omdel
    """
    gpu = max(1, gpu)
    # Gather input information for building a model
    input_height = X_train[0].shape[0]
    input_width = X_train[0].shape[1]
    input_channels = X_train[0].shape[2]
    output_channels = Y_train[0].shape[2]

    if load_model_path is not None:
        model = load_models(load_model_path, gpu)[0]
    else:
        model = build_model(input_height, input_width, input_channels, output_channels, min_dropout)

     # A multi-GPU model or a normal model depending on the number of GPU
    typed_model = get_typed_model(model, gpu)
    scores = []
    # train the model
    if (len(X_val) > 0) and (len(Y_val) > 0):
        earlystopper = EarlyStopping(patience=earlystopping, verbose=0)
        typed_model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=epochs, batch_size=int(gpu*16), callbacks=[earlystopper], verbose=1)
        scores = evaluate_model(typed_model, X_val,  Y_val)
    else:
        typed_model.fit(X_train, Y_train, epochs=epochs, batch_size=int(gpu*16), verbose=1)

    save_model(model, 'single-', save_model_path)
   

    return model, scores
    

def train_model_kfold(X_train, Y_train, n_splits, epochs=20, gpu=1, earlystopping=5, load_models_path=None, save_model_path=None, min_dropout=0.1):
    """ Build or load K models and train data on the K models.  
    """
    # the number of gpu determins the batch size, if it's 0, we set it to 1
    gpu = max(1, gpu)
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    # store cvscores and models
    cvscores = []
    models = []
    pretrained_models = []
    
    if load_models_path is not None:
        pretrained_models = load_models(load_models_path, gpu)
    
    # Gather input information for building a model
    input_height = X_train[0].shape[0]
    input_width = X_train[0].shape[1]
    input_channels = X_train[0].shape[2]
    output_channels = Y_train[0].shape[2]
    
    for index, (train, val) in enumerate(kfold.split(X_train)):
        # keep the non-GPU model type for saving model due to the Keras restriction
        if not pretrained_models:
            model = build_model(input_height, input_width, input_channels, output_channels, min_dropout)
        else:
            model = pretrained_models[index]

        # A multi-GPU model or a normal model depending on the number of GPU
        typed_model = get_typed_model(model, gpu)
        # fit the model
        earlystopper = EarlyStopping(monitor="val_loss", patience=earlystopping, verbose=0)
        typed_model.fit(X_train[train], Y_train[train], validation_data=[X_train[val], Y_train[val]], epochs=epochs, batch_size=int(gpu*16), callbacks=[earlystopper], verbose=1)
        save_model(model, str(n_splits)+'-fold-', save_model_path)
        # evaluate the model
        scores = evaluate_model(typed_model, X_train[val], Y_train[val])
        models.append(typed_model)
        cvscores.append([scores[0] * 100, scores[1] * 100])

    val_metric_means = np.asarray(cvscores).mean(axis=0)
    val_metric_std = np.asarray(cvscores).std(axis=0)
    print("%s: %.2f%% (+/- %.2f%%)" % (typed_model.metrics_names[0], val_metric_means[0], val_metric_std[0]))
    print("%s: %.2f%% (+/- %.2f%%)" % (typed_model.metrics_names[1], val_metric_means[1], val_metric_std[1]))
    
    return models, cvscores


def evaluate_model(model, X_val, Y_val):
    """evaluate the model"""
    scores = model.evaluate(X_val, Y_val, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[0], scores[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    return scores


def load_models(model_path, gpu=1):
    models = []
    (_, _, model_ids) = next(os.walk(model_path))
    if not model_ids:
        raise ValueError('Empty directory ' + model_path + ' reading trained models.')

    for id_ in sorted(model_ids):
        full_model_path = os.path.join(model_path, id_)
        print("Loading model: " + full_model_path)
        model = load_model(full_model_path, custom_objects={'dice_coef': dice_coef})
        model = get_typed_model(model, gpu)
        models.append(model)

    return models
