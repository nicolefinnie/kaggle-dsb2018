import os
import sys
import math
import datetime
import logging
from logging.config import fileConfig
import argparse
from keras import backend as K

"""Import our own modules"""
sys.path.append("modules")
from modules.augment import *
from modules.mean_IoU import *
from modules.image_io import *
from modules.image_resize import *
from modules.image_processing import *
from modules.submission import *
from modules.prediction import *
from modules.unet import *
import modules.image_mosaic as mosaic
import modules.image_processing as impr


class NucleiUtility(object):
    def __init__(self): 
        self.OUTPUT_PATH = 'output_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        self.MODELS_GREY_PATH = 'models_grey'
        self.MODELS_COLOUR_PATH = 'models_colour'
        self.LOAD_MODELS_GREY_PATH = ''
        self.LOAD_MODELS_COLOUR_PATH = ''
        
        self.PREDICT_TRAIN_GREY_PATH = 'predict/train/grey'
        self.PREDICT_TRAIN_COLOUR_PATH = 'predict/train/colour'
        self.PREDICT_VAL_GREY_PATH = 'predict/val/grey'
        self.PREDICT_VAL_COLOUR_PATH = 'predict/val/colour'
        self.PREDICT_TEST_GREY_PATH = 'predict/test/grey'
        self.PREDICT_TEST_COLOUR_PATH = 'predict/test/colour'
        self.IMG_WIDTH = 256
        self.IMG_HEIGHT = 256
       
        """============Tunable parameters from the command line==========="""
        self.TRAIN_PATH = '../data/stage1_train/'
        self.TEST_PATH = '../data/stage1_test/'
        self.RETRAIN = False     
        self.GPU = 1
        self.EPOCH = 1
        self.KFOLD = -1
        self.VALSPLIT = 0
        self.EARLYSTOPPING = 5      
        self.WEIGHTS = []

        self.GREY_ONLY = False
        self.COLOUR_ONLY = False
        self.PREDICT_TEST_IMAGES_ONLY = False
        self.VISUALIZE = False
        self.MOSAIC = False

        ### Hyper parameters ###
        self.DROPOUT = 0.1

        ### Data augmentation parameters: sigma, scale, alpha ###
        self.MAX_TRAIN_SIZE = 150000
        self.ROTATE_IMAGES = True
        self.INVERT_IMAGES = False
        self.PYRAMID_SCALE = -1
        self.BLUR_SIGMA = -1
        self.TRANSFORM_SIGMA = -1
        self.NOISE_SCALE = -1
        self.GREYSCALE_ALPHA = -1

        """ Indices of grey/colour clusters, these will be set automatically 
         after the pipeline detects the colour of images in the cluster """
        self.GREY_IX = 0
        self.COLOUR_IX = 1     


        """================================================================="""
        try:
            fileConfig('logging.conf')
        except:
            pass    
        self.logger = logging.getLogger(__name__)
        self.logger.debug('================================Nuclei START==================================')

    def parse_argument(self, arguments=None):
        """Parse the arguments from the command line"""
        is_parser_error = False
        parser = argparse.ArgumentParser(description='This tool helps to train and predict nuclei segmentation.', 
                                        formatter_class=argparse.RawTextHelpFormatter)
        
        parser.add_argument('action', type=str,
                            choices=['train', 'predict', 'loadpredict'],
                            help='train - training the model\n'+
                                'predict - predict test data\n'+
                                'loadpredict - load predictions\n' +
                                'if --trainpath or --testpath or --predictpath is specified, we use the path\n')

        parser.add_argument('--gpu', nargs=1, metavar='<number of GPUs>',
                            help='the number of GPUs to train, predict and decide the batch size.' + 
                            'If not specified, GPU is 1 by default.') 
        
        parser.add_argument('--trainpath', nargs=1, metavar='<training data set path>',
                            help='Specify the file path of the training data set') 
        parser.add_argument('--testpath', nargs=1, metavar='<test data set path>',
                            help='Specify the file path of the test data set') 
        parser.add_argument('--loadmodel',nargs=1, metavar='<path of pretrained models>', 
                            help='Specify the path of pretrained models')                            
        parser.add_argument('--predictpath',nargs=1, metavar='<path of all predictions>', 
                            help='Specify the path of predictions on training / testing images')
        parser.add_argument('--weights', nargs='*', dest='weights',
                            help='Specify the weights of all loaded models')

        parser.add_argument('--kfold', type=int, nargs=1, metavar='<k-fold of training model>',
                            help='k: the number of splitting the training data set to k-fold models.' +
                            'If not specified, K is 2 by default.') 
        parser.add_argument('--valsplit', type=float, nargs=1, metavar='<validation set split of a single training model>',
                            help='The given percentage split from the training set used for validation set.' +
                            'If not specified, K is 2 by default.')                     
        parser.add_argument('--epoch', type=int, nargs=1, metavar='<epochs of training>',
                            help='How many epochs we want to train for the k-fold model.' + 
                            'If not specified, EPOCH=1 by default.') 
        parser.add_argument('--earlystopping', type=int, nargs=1, metavar='<epochs of non-decreasing validation loss>',
                            help='After how many epochs do we want to stop trainning after the validation loss has stopped decreasing.' + 
                            'If not specified, EARLYSTOPPING=5 by default.') 

        parser.add_argument('--greyonly', action="store_true",
                            help='Set: only grey models will be trained and does predictions on grey images')     
        parser.add_argument('--colouronly', action="store_true", 
                            help='Set: only colour models will be trained and does predictions on colour images')    

        parser.add_argument('--maxtrainsize', type=int, nargs=1, metavar='<approximate maximum training images>',
                            help='Augmentation can significantly increase the number of training images. This parameter ' +
                            'allows you to limit the total training images.') 

        parser.add_argument('--dropout', type=float, nargs=1, metavar='<minimum dropout rate, such as 0.1>', 
                            help='the minimum dropout rate for the model') 
        parser.add_argument('--rotate', action="store_true", 
                            help='Set: training data will be augmented by rotating\n'+
                                'Not set: training data will not be augmented by rotating')
        parser.add_argument('--scale', type=float, nargs=1, metavar='<scale, such as 1.5>', 
                            help='Training data will be scaled using the given scale based on the average nuclei size')      
        parser.add_argument('--blur', type=float, nargs=1, metavar='<sigma, such as 1>', 
                            help='Training data will be blurred using the given sigma')      
        parser.add_argument('--transform', type=float, nargs=1, metavar='<sigma, such as 0.1>', 
                            help='Training data will be perspectively transformed using the given sigma')      
        parser.add_argument('--noise', type=float, nargs=1, metavar='<scale, such as 0.05>', 
                            help='Add noise to the training data using the given scale')    
        parser.add_argument('--greyscale', type=float, nargs=1, metavar='<alpha, such as 1>', 
                            help='Add greyscale images to the training data using the given alpha')    
        parser.add_argument('--invert', action="store_true", 
                            help='Set: add inverted imaged to the training')    


        parser.add_argument('--visualize', action="store_true", 
                            help='Set: visualizing and ploting desired images')    
        parser.add_argument('--mosaic', action="store_true", 
                            help='Set: Try to form mosaics from input images')    
        parser.add_argument('--predicttestonly', action="store_true", 
                            help='Set: only predict on test data set')
     
                

        try:
            if arguments is None:
                args = parser.parse_args()
            else:
                args = parser.parse_args(arguments)
            
            if args.trainpath is not None:
                if not os.path.isdir(args.trainpath[0]):
                    is_parser_error = True
                    self.logger.error('The training data path \'' + args.trainpath[0] + '\' you specified does not exist\n')
                else:
                    self.TRAIN_PATH = args.trainpath[0]
            if args.testpath is not None: 
                if not os.path.isdir(args.testpath[0]):
                    is_parser_error = True
                    self.logger.error('The test data path \'' + args.testpath[0] + '\' you specified does not exist\n')
                else:
                    self.TEST_PATH = args.testpath[0]
            if args.loadmodel is not None:
                if not os.path.isdir(args.loadmodel[0]):
                    is_parser_error = True
                    self.logger.error('The loading model path \'' + args.loadmodel[0] + '\' you specified does not exist\n')
                else:
                    self.LOAD_MODELS_COLOUR_PATH = os.path.join(args.loadmodel[0],  self.MODELS_COLOUR_PATH)
                    self.LOAD_MODELS_GREY_PATH = os.path.join(args.loadmodel[0],  self.MODELS_GREY_PATH)

            if args.weights is not None:
                for arg in args.weights:
                    self.WEIGHTS.append(float(arg)) 

            if args.predictpath is not None:
                if not os.path.isdir(args.predictpath[0]):
                    is_parser_error = True
                    self.logger.error('The prediction path \'' + args.predictpath[0] + '\' you specified does not exist\n')
                else:
                    self.logger.debug('The prediction path is %s', args.predictpath[0])
                    self.PREDICT_TRAIN_GREY_PATH = os.path.join(args.predictpath[0], self.PREDICT_TRAIN_GREY_PATH)
                    self.PREDICT_TRAIN_COLOUR_PATH = os.path.join(args.predictpath[0], self.PREDICT_TRAIN_COLOUR_PATH)
                    self.PREDICT_VAL_GREY_PATH = os.path.join(args.predictpath[0], self.PREDICT_VAL_GREY_PATH)
                    self.PREDICT_VAL_COLOUR_PATH = os.path.join(args.predictpath[0], self.PREDICT_VAL_COLOUR_PATH)
                    self.PREDICT_TEST_GREY_PATH = os.path.join(args.predictpath[0], self.PREDICT_TEST_GREY_PATH)
                    self.PREDICT_TEST_COLOUR_PATH = os.path.join(args.predictpath[0], self.PREDICT_TEST_COLOUR_PATH)
                    

            if args.gpu is not None:
                self.GPU = args.gpu[0]
            if args.epoch is not None:
                self.EPOCH = args.epoch[0]
            if args.kfold  is not None:
                self.KFOLD = args.kfold[0]
            if args.valsplit is not None:
                self.VALSPLIT = args.valsplit[0]
            if args.earlystopping is not None:
                self.EARLYSTOPPING = args.earlystopping[0]
            if args.dropout is not None:
                self.DROPOUT = args.dropout[0]
                   
            if args.maxtrainsize is not None:
                self.MAX_TRAIN_SIZE = args.maxtrainsize[0]

            if args.scale is not None:
                self.PYRAMID_SCALE = args.scale[0]
            if args.blur is not None:
                self.BLUR_SIGMA = args.blur[0]
            if args.transform is not None:
                self.TRANSFORM_SIGMA = args.transform[0]
            if args.noise is not None:
                self.NOISE_SCALE = args.noise[0]
            if args.greyscale is not None:
                self.GREYSCALE_ALPHA = args.greyscale[0]
            
            self.ROTATE_IMAGES = args.rotate
            self.INVERT_IMAGES = args.invert

            self.VISUALIZE = args.visualize
            self.MOSAIC = args.mosaic
            self.PREDICT_TEST_IMAGES_ONLY = args.predicttestonly
            self.GREY_ONLY = args.greyonly
            self.COLOUR_ONLY = args.colouronly

            self.logger.debug(args)
        except:
            is_parser_error = True
            
        if is_parser_error is True:    
            parser.print_help()
            sys.exit(1)
        else:
            return args
        

    def test_gpu(self):
        found_gpus = K.tensorflow_backend._get_available_gpus()
        self.logger.info("Found GPUs: " + str(found_gpus))


    def load_train_data(self, num_cluster=2):
        self.logger.info('Loading train images as a data frame from %s', self.TRAIN_PATH)
        img_df = read_images_as_dataframe(self.TRAIN_PATH)
        self.logger.info('Loading train masks and contours to the data frame and save the masks/contours on the disk')
        read_masks_to_dataframe(self.TRAIN_PATH, img_df)
        img_df = create_color_features(img_df)
        img_df, cluster_maker = create_color_clusters(img_df, num_cluster)
        
        self.logger.debug('Original training data set size: %d', len(img_df))

        return img_df, cluster_maker


    def load_test_data(self, cluster_maker, num_cluster = 2):
        self.logger.info('Loading test images as a data frame from %s', self.TEST_PATH)
        img_df = read_images_as_dataframe(self.TEST_PATH)
        img_df =  create_color_features(img_df)
        img_df, _ = create_color_clusters(img_df, num_cluster, cluster_maker)
        
        self.logger.debug('Original test data set size: %d', len(img_df))

        return img_df


    def set_cluster_index(self, cluster):
        count = 0
        for _, row in cluster[0].iterrows():
            if row['Red'] == row['Green'] and row['Green'] == row['Blue']:
                count = count+1
        if count > float(len(cluster[0].index)/2):
            self.logger.info('After K-Means: set the cluster 0 to GREY and the cluster 1 to COLOUR ')
            self.GREY_IX = 0
            self.COLOUR_IX = 1
        else:
            self.logger.info('After K-Means: set the cluster 0 to COLOUR and the cluster 1 to GREY ')
            self.GREY_IX = 1
            self.COLOUR_IX = 0


    def preprocess_images(self, train_img_df, test_img_df, num_clusters = 2):
        process_images(train_img_df)
        process_images(test_img_df) 
            
        cluster_train_df_list = split_cluster_to_group(train_img_df, num_clusters)
        cluster_test_df_list = split_cluster_to_group(test_img_df, num_clusters)
        
        self.set_cluster_index(cluster_train_df_list)

        self.logger.debug('Grey train images size: %d', len(cluster_train_df_list[self.GREY_IX]))
        self.logger.debug('Colour train images size: %d', len(cluster_train_df_list[self.COLOUR_IX]))
        
        self.logger.debug(cluster_train_df_list[0].sample())
        self.logger.debug(cluster_test_df_list[0].sample())
        return cluster_train_df_list, cluster_test_df_list


    def augment_training_inputs(self, X_train, Y_train, X_train_3channel):
        augmented_X_train = X_train.copy()
        augmented_Y_train = Y_train.copy()

        if self.PYRAMID_SCALE > 0:
            self.logger.info('Scale train images based on the nuclei size')
            nuclei_sizes = get_mean_cell_size(Y_train)
            self.logger.info('Median nuclei size of train images: %d', np.median(nuclei_sizes))
            augmented_X_train.extend(scale_images_on_nuclei_size(X_train, nuclei_sizes, self.PYRAMID_SCALE, self.IMG_HEIGHT/2, self.IMG_HEIGHT*2))
            augmented_Y_train.extend(scale_images_on_nuclei_size(Y_train, nuclei_sizes, self.PYRAMID_SCALE, self.IMG_HEIGHT/2, self.IMG_HEIGHT*2))
           
        if self.TRANSFORM_SIGMA > 0:
            self.logger.info('Perspectively transform train images')
            sequence = get_perspective_transform_sequence(self.TRANSFORM_SIGMA)
            augmented_X_train.extend(perspective_transform(X_train, sequence))
            augmented_Y_train.extend(perspective_transform(Y_train, sequence))
            if len(X_train_3channel) > 0:
               self.logger.info('Perspectively transform train images a second time on colour images')
               sequence = get_perspective_transform_sequence(self.TRANSFORM_SIGMA - 0.005)
               augmented_X_train.extend(perspective_transform(X_train, sequence))
               augmented_Y_train.extend(perspective_transform(Y_train, sequence))

        if self.NOISE_SCALE > 0:
            self.logger.info('Add additive Gaussian noise and speckle noise to train images')
            augmented_X_train.extend(additive_Gaussian_noise(X_train, self.NOISE_SCALE))
            augmented_X_train.extend(speckle_noise(X_train))
            self.logger.debug('We do not add noise to mask data, just add the original mask data')
            augmented_Y_train.extend(Y_train)
            augmented_Y_train.extend(Y_train)
                         
        if self.GREYSCALE_ALPHA > 0 and len(X_train_3channel) > 0:
            self.logger.info('Convert images to greyscale on original RGB images')
            augmented_X_train.extend(greyscale(X_train_3channel, self.GREYSCALE_ALPHA))
            self.logger.debug('We do not greysacle on mask data, just add the original mask data')
            augmented_Y_train.extend(Y_train)
                
        if self.INVERT_IMAGES:
            self.logger.info('Invert images on original 3-channel images')
            augmented_X_train.extend(invert(X_train))
            self.logger.debug('We do not invert mask data, just add the original mask data')
            augmented_Y_train.extend(Y_train)
              
        if self.BLUR_SIGMA > 0:
            self.logger.info('Blurring train images')
            augmented_X_train.extend(blur(X_train, self.BLUR_SIGMA))
            self.logger.debug('We do not blur on mask data, just add the original mask data')
            augmented_Y_train.extend(Y_train)

        # Best results where when the output from windowing was rotated, instead of re-sizing here
        # and only rotating on some data. So, rotation is done afterwards.
        #if self.ROTATE_IMAGES:
        #    self.logger.info('Rotate and mirror train images')
        #    augmented_X_train.extend( augment_max ( np.asarray( resize_images(X_train, self.IMG_HEIGHT) ) ) )
        #    augmented_Y_train.extend( augment_max ( np.asarray( resize_images(Y_train, self.IMG_HEIGHT) ) ) )

        self.logger.info('Augmentation rate is %d', len(augmented_X_train)/len(X_train))

        return (augmented_X_train, augmented_Y_train)


    def build_model_training_inputs(self, cluster_df, cluster_ix):
        X_train_3channel = []
        all_proc_images = cluster_df['image_process'].values.tolist()
        num_orig_images = len(all_proc_images)
        all_masks = cluster_df['mask_train'].values.tolist()
        if self.MOSAIC:
            self.logger.info('Forming mosaics from %d input images...', len(all_proc_images))
            # Mosaics are formed on the raw image, since individual parts of a mosaic may have
            # been altered differently during pre-processing.
            mosaic_images, _, mosaic_dict, not_combined = mosaic.make_mosaic(cluster_df['image'].values.tolist(), None)
            self.logger.info('Found %d 4x4 image mosaics, %d images could not be combined into mosaics.', len(mosaic_images), len(not_combined))
            self.logger.debug('Mosaic dictionary: %s', mosaic_dict)
            self.logger.debug('Images that could not be combined into mosaics: %s', str(not_combined))
            
            # Augmentation needs the original 3-channel colour image in some cases too
            if cluster_ix == self.COLOUR_IX:
                (X_train_3channel, _, _) = mosaic.merge_mosaic_images(mosaic_dict, mosaic_images, cluster_df['image'].values.tolist())

            mosaic_images = [impr.preprocess_image(x) for x in mosaic_images]
            (X_train, _, Y_train) = mosaic.merge_mosaic_images(mosaic_dict, mosaic_images, all_proc_images, all_masks)
            self.logger.info('Total of %d images after mosaic processing.', len(X_train))
            mosaic_images = None
        else:
            X_train = all_proc_images
            Y_train = all_masks
            if cluster_ix == self.COLOUR_IX:
                X_train_3channel = cluster_df['image'].values.tolist()

        self.logger.info('%d images of the original training data', len(X_train))
        
        if len(X_train) > 0:
            (X_train, Y_train) = self.augment_training_inputs(X_train, Y_train, X_train_3channel)
        X_train_3channel = None

        self.logger.info('Windowing on training data')
        X_train = window_images(X_train, self.IMG_HEIGHT, self.IMG_WIDTH)
        Y_train = window_images(Y_train, self.IMG_HEIGHT, self.IMG_WIDTH)
        self.logger.info('%d images to the training data after windowing', X_train.shape[0])

        # Rotations/flips moved here instead of in the main augmentation loop, to ensure all
        # augmented samples are also mirrored/flipped.
        if self.ROTATE_IMAGES and len(X_train) > 0:
            self.logger.info('Rotate and mirror train images')
            rotate_amplify_rate = 8
            num_windows = X_train.shape[0]
            estimated_images = num_windows * rotate_amplify_rate
            if estimated_images > self.MAX_TRAIN_SIZE:
                max_windows_to_rotate = int(self.MAX_TRAIN_SIZE/rotate_amplify_rate)
                self.logger.info('Only rotating the first %d windows to reduce training size.', max_windows_to_rotate)
                augment_half_X = augment_max(X_train[0:max_windows_to_rotate])
                X_train = np.concatenate((augment_half_X, X_train[max_windows_to_rotate:]), axis=0)
                augment_half_X = None
                augment_half_Y = augment_max(Y_train[0:max_windows_to_rotate])
                Y_train = np.concatenate((augment_half_Y, Y_train[max_windows_to_rotate:]), axis=0)
                augment_half_Y = None
            else:
                X_train = augment_max(X_train)
                Y_train = augment_max(Y_train)
            self.logger.info('%d images to the training data after rotations/flips', X_train.shape[0])

        if len(X_train) > 0:
            self.logger.info('Final augmentation rate is %d', int(X_train.shape[0]/num_orig_images))

        return (X_train, Y_train)


    def build_model_prediction_inputs(self, images):
        inputs = window_images(images, self.IMG_HEIGHT, self.IMG_WIDTH)
        inputs = augment_max(inputs)
        self.logger.info('%d images in prediction data after windowing/rotation/flip', len(inputs))

        return inputs


    def train_model(self, train_df, val_df, cluster_ix, save_model_path, load_model_path=None, model_type='grey'):
        self.logger.info('### Build X_train/Y_train %s: images ####', model_type)
        (X_train, Y_train) = self.build_model_training_inputs(train_df, cluster_ix)

        if self.KFOLD > 0:
            if os.path.isdir(load_model_path):
                self.logger.info('Retrain the %s models under %s and save the new models under %s', model_type, load_model_path, save_model_path)
                models, _ = train_model_kfold(X_train, Y_train, self.KFOLD, self.EPOCH, self.GPU, self.EARLYSTOPPING, load_model_path, save_model_path, min_dropout=self.DROPOUT)
            else:
                self.logger.info("Train the new %s models from scratch and save the new models under %s", model_type, save_model_path)
                models, _ = train_model_kfold(X_train, Y_train, self.KFOLD, self.EPOCH, self.GPU, self.EARLYSTOPPING, save_model_path=save_model_path, min_dropout=self.DROPOUT)

        else:
            self.logger.info('### Build X_val/Y_val %s: images ####', model_type)
            (X_val, Y_val) = self.build_model_training_inputs(val_df, cluster_ix)
            if os.path.isdir(load_model_path):
                self.logger.info('Retrain the single %s model under %s and save the new model under %s', model_type, load_model_path, save_model_path)
                model, _ = train_model(X_train, Y_train, X_val, Y_val, self.EPOCH, self.GPU, self.EARLYSTOPPING, load_model_path, save_model_path, min_dropout=self.DROPOUT)
            else:
                self.logger.info("Train the new %s model from scratch and save the new model under %s", model_type, save_model_path)
                model, _ = train_model(X_train, Y_train, X_val, Y_val, self.EPOCH, self.GPU, self.EARLYSTOPPING, save_model_path=save_model_path, min_dropout=self.DROPOUT)

            models = [model]
           
        return models


    def build_train_validation_df_list(self, cluster_train_df_list):
        self.logger.info('Remove outliers ONLY from the training data fed to the model')
        no_outliers_train_df_list = []
        no_outliers_train_df_list.append(return_images_without_outliers(cluster_train_df_list[0]))
        no_outliers_train_df_list.append(return_images_without_outliers(cluster_train_df_list[1]))
        self.logger.debug('Grey train images size without outliers: %d', len(no_outliers_train_df_list[self.GREY_IX]))
        self.logger.debug('Colour Train images size without outliers: %d', len(no_outliers_train_df_list[self.COLOUR_IX]))

        train_df_list = []
        val_df_list = []
        train_df_0, val_df_0 = split_train_val_set(no_outliers_train_df_list[0], self.VALSPLIT)
        train_df_1, val_df_1 = split_train_val_set(no_outliers_train_df_list[1], self.VALSPLIT)

        train_df_list.append(train_df_0)
        val_df_list.append(val_df_0)
        train_df_list.append(train_df_1)
        val_df_list.append(val_df_1)

        self.logger.debug('sorted validation 0 imageID:\n%s', str(np.sort(val_df_0['imageID'].values)))
        self.logger.debug('sorted validation 1 imageID:\n%s', str(np.sort(val_df_1['imageID'].values)))
        return train_df_list, val_df_list

    def train(self, train_df_list, val_df_list):
        save_model_grey_path = os.path.join(self.OUTPUT_PATH, self.MODELS_GREY_PATH) 
        save_model_colour_path = os.path.join(self.OUTPUT_PATH, self.MODELS_COLOUR_PATH)
        
        grey_models = []
        colour_models = []

        if not self.COLOUR_ONLY:
            grey_models = self.train_model(train_df_list[self.GREY_IX], val_df_list[self.GREY_IX], self.GREY_IX, save_model_grey_path, self.LOAD_MODELS_GREY_PATH, 'grey')
        if not self.GREY_ONLY:
            colour_models = self.train_model(train_df_list[self.COLOUR_IX], val_df_list[self.COLOUR_IX], self.COLOUR_IX, save_model_colour_path, self.LOAD_MODELS_COLOUR_PATH, 'colour')

        return grey_models, colour_models

    def batched_predictions(self, img_df, models, input_type):
        # Iterator to chunk images/sizes into batches.
        def image_batches(images, sizes, batch_size):
            for i in range(0, len(images), batch_size):
                yield (images[i:i + batch_size], sizes[i:i + batch_size])

        predictions_full_size = []
        all_images = img_df['image_process'].values.tolist()
        if self.MOSAIC:
            self.logger.info('Forming mosaics from %d input %s images...', len(all_images), input_type)
            # Mosaics are formed on the raw image, since individual parts of a mosaic may have
            # been altered differently during pre-processing.
            mosaic_images, _, mosaic_dict, not_combined = mosaic.make_mosaic(img_df['image'].values.tolist(), None)
            mosaic_images = [impr.preprocess_image(x) for x in mosaic_images]
            self.logger.info('Found %d 4x4 image mosaics, %d images could not be combined into mosaics.', len(mosaic_images), len(not_combined))
            self.logger.debug('Mosaic dictionary: %s', mosaic_dict)
            self.logger.debug('Images that could not be combined into mosaics: %s', str(not_combined))
            
            # Any images not included in the mosaic images should be from the list of pre-processed images
            (all_images, all_sizes, _) = mosaic.merge_mosaic_images(mosaic_dict, mosaic_images, all_images)
            self.logger.info('Total of %d images after mosaic processing.', len(all_images))
            mosaic_images = None
        else:
            all_sizes = img_df['size'].values

        # Split the total set of images into smaller batches. For large datasets (i.e. after applying
        # windowing and rotation), trying to do all images at once encountered a Python MemoryError.
        batch_size = 100
        self.logger.info('Predict on %s images in batches of up to %d original images...', input_type, batch_size)
        for (batch, sizes) in image_batches(all_images, all_sizes, batch_size):
            predict_inputs = self.build_model_prediction_inputs(batch)
            predictions = average_model_predictions(predict_inputs, models, self.WEIGHTS)
            predictions_full_size.extend(predict_restore_to_fullsize(predictions, sizes))
            del predictions

        if self.MOSAIC:
            self.logger.info('Re-forming full-size images from mosaics.')
            input_len = len(img_df['image_process'].values.tolist())
            predictions_full_size = mosaic.split_merged_mosaic(mosaic_dict, predictions_full_size, input_len)

        return predictions_full_size

    def predict_model_for_dataframe(self, img_df, models, predict_path, input_type, colour_type):
            self.logger.info('Performing batched %s predictions for %s images...', input_type, colour_type)
            predictions_full_size = self.batched_predictions(img_df, models, input_type)
            
            img_df['prediction'] = pd.Series(predictions_full_size).values
            predictions_full_size = None

            self.logger.info('Save predictions of %s %s images under %s', input_type, colour_type, os.path.join(self.OUTPUT_PATH, predict_path))
            save_prediction_images(img_df, os.path.join(self.OUTPUT_PATH, predict_path))

            self.post_process_predictions(img_df, input_type, colour_type)

    def predict_model(self, grey_models, colour_models, cluster_df_list, input_type='train'):
        if input_type == 'train':
            predict_grey_path = self.PREDICT_TRAIN_GREY_PATH
            predict_colour_path = self.PREDICT_TRAIN_COLOUR_PATH
        elif input_type == 'val':
            predict_grey_path = self.PREDICT_VAL_GREY_PATH
            predict_colour_path = self.PREDICT_VAL_COLOUR_PATH
        elif input_type == 'test':
            predict_grey_path = self.PREDICT_TEST_GREY_PATH
            predict_colour_path = self.PREDICT_TEST_COLOUR_PATH
            
        if grey_models:
            self.predict_model_for_dataframe(cluster_df_list[self.GREY_IX], grey_models, predict_grey_path, input_type, 'grey')

        if colour_models:
            self.predict_model_for_dataframe(cluster_df_list[self.COLOUR_IX], colour_models, predict_colour_path, input_type, 'colour')

    def predict(self, cluster_train_df_list, cluster_val_df_list, cluster_test_df_list, grey_models=None, colour_models=None):
        if grey_models is None and self.LOAD_MODELS_GREY_PATH:
            grey_models = load_models(self.LOAD_MODELS_GREY_PATH, self.GPU)
        if colour_models is None and self.LOAD_MODELS_COLOUR_PATH:
            colour_models = load_models(self.LOAD_MODELS_COLOUR_PATH, self.GPU)

        if not self.PREDICT_TEST_IMAGES_ONLY:
            self.predict_model(grey_models, colour_models, cluster_train_df_list, 'train')

        if not cluster_val_df_list[self.GREY_IX].empty and not cluster_val_df_list[self.COLOUR_IX].empty:
            self.predict_model(grey_models, colour_models, cluster_val_df_list, 'val')

        self.predict_model(grey_models, colour_models, cluster_test_df_list, 'test')

        self.logger.debug('Sample the final test data frame after prediction to make sure everything is good')
        self.logger.debug(cluster_test_df_list[self.GREY_IX].sample())
        self.logger.debug(cluster_test_df_list[self.COLOUR_IX].sample())

    def post_process_predictions(self, img_df, input_type, colour_type):
        self.logger.info('Post processing predicted %s %s images and generate labels for each image', input_type, colour_type)
        if self.MOSAIC:
            # Mosaic prediction steps:
            # 1. Make the list of mosaic images
            # 2. Merge the mosaic images into the list of all images (needed for post-processing), and at the
            #    same time, update the predictions so they match the mosaic images if needed.
            # 3. Perform post-processing to get the output labels (one label for each mosaic image)
            # 4. Split the mosaic labels to match the original images
            # 5. Re-label the split labels, to ensure all nuclei in a single label image are consecutive
            all_images = img_df['image_process'].values.tolist()
            all_predictions = img_df['prediction'].values.tolist()
            input_len = len(all_images)
            mosaic_images, _, mosaic_dict, _ = mosaic.make_mosaic(img_df['image'].values.tolist(), None)
            mosaic_images = [impr.preprocess_image(x) for x in mosaic_images]
            (all_images, _, all_predictions) = mosaic.merge_mosaic_images(mosaic_dict, mosaic_images, all_images, all_predictions)
            labels = [impr.post_process_image(img, prediction[:,:,0], prediction[:,:,1]) for img, prediction in zip(all_images, all_predictions)]
            labels = mosaic.split_merged_mosaic(mosaic_dict, labels, input_len)
            labels = [renumber_labels(label_img) for label_img in labels]
            img_df['label'] = pd.Series(labels).values
        else:
            add_labels_to_dataframe(img_df)


    def load_predictions_by_group(self, cluster_df_list, grey_path, colour_path, input_type):
        if os.path.isdir(grey_path):
            self.logger.info('Load predictions of %s grey images...', input_type)
            load_prediction_images_to_df(cluster_df_list[self.GREY_IX], grey_path)
            self.post_process_predictions(cluster_df_list[self.GREY_IX], input_type, 'grey')

        if os.path.isdir(colour_path):
            self.logger.info('Load predictions of %s colour images...', input_type)
            load_prediction_images_to_df(cluster_df_list[self.COLOUR_IX], colour_path)
            self.post_process_predictions(cluster_df_list[self.COLOUR_IX], input_type, 'colour')


    def load_predictions(self, cluster_train_df_list, cluster_val_df_list, cluster_test_df_list):
        self.load_predictions_by_group(cluster_train_df_list, self.PREDICT_TRAIN_GREY_PATH, self.PREDICT_TRAIN_COLOUR_PATH, 'training')
        self.load_predictions_by_group(cluster_val_df_list, self.PREDICT_VAL_GREY_PATH, self.PREDICT_VAL_COLOUR_PATH, 'validation')
        self.load_predictions_by_group(cluster_test_df_list, self.PREDICT_TEST_GREY_PATH, self.PREDICT_TEST_COLOUR_PATH, 'test')
        

    def create_submission(self, cluster_test_df_list):
        if 'label' in cluster_test_df_list[self.GREY_IX] and 'label' in cluster_test_df_list[self.COLOUR_IX]:
            submission_file = os.path.join(self.OUTPUT_PATH, str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '.csv')
            self.logger.info('Generate submission file')
            dfs = []
            dfs.append(cluster_test_df_list[self.GREY_IX])
            dfs.append(cluster_test_df_list[self.COLOUR_IX])
            generate_submission_from_df(dfs, submission_file)
            self.logger.info("##### File ready for submission: %s ######", submission_file)
        else:
            self.logger.info('No submission file will be generated because not all images have been predicted.')


    def display_mean_IoU(self, df):
        self.logger.info('Mean IoU: ' + str(df['mean_IoU'].mean()))
        self.logger.info('Sorted mean IoUs for 100-image groups')
        self.logger.info(df.sort_values(by=['mean_IoU'])['mean_IoU'].groupby(np.arange(len(df))//100).mean())


    def generate_meanIoU_report(self, cluster_split_train_df_list, cluster_val_df_list):
        self.logger.info('Generate ground truths for mean IoU on training and validation data....')
        Y_true = create_labeled_masks(self.TRAIN_PATH)
       
        if 'label' in cluster_split_train_df_list[self.GREY_IX]:
            self.logger.info('Calculate mean IoU on grey training data:')
            cluster_split_train_df_list[self.GREY_IX] = add_metrics_to_df(cluster_split_train_df_list[self.GREY_IX], Y_true, True)
            self.display_mean_IoU(cluster_split_train_df_list[self.GREY_IX])

        if 'label' in cluster_split_train_df_list[self.COLOUR_IX]:
            self.logger.info('Calculate mean IoU on colour training data:')
            cluster_split_train_df_list[self.COLOUR_IX] = add_metrics_to_df(cluster_split_train_df_list[self.COLOUR_IX], Y_true, True)
            self.display_mean_IoU(cluster_split_train_df_list[self.COLOUR_IX])

        if 'label' in cluster_val_df_list[self.GREY_IX]:
            self.logger.info('Calculate mean IoU on grey validation data:')
            cluster_val_df_list[self.GREY_IX] = add_metrics_to_df(cluster_val_df_list[self.GREY_IX], Y_true, True)
            self.display_mean_IoU(cluster_val_df_list[self.GREY_IX])

        if 'label' in cluster_val_df_list[self.COLOUR_IX]:
            self.logger.info('Calculate mean IoU on colour validation data:')
            cluster_val_df_list[self.COLOUR_IX] = add_metrics_to_df(cluster_val_df_list[self.COLOUR_IX], Y_true, True)
            self.display_mean_IoU(cluster_val_df_list[self.COLOUR_IX])



if __name__ == '__main__':
    util = NucleiUtility()
    args = util.parse_argument()

    util.test_gpu()
    train_img_df, cluster_maker = util.load_train_data()
    test_img_df = util.load_test_data(cluster_maker)
    cluster_train_df_list, cluster_test_df_list = util.preprocess_images(train_img_df, test_img_df) 

    cluster_split_train_df_list, cluster_val_df_list = util.build_train_validation_df_list (cluster_train_df_list) 
    if args.action == 'train':
        grey_models, colour_models = util.train(cluster_split_train_df_list, cluster_val_df_list)
        util.predict(cluster_split_train_df_list, cluster_val_df_list, cluster_test_df_list, grey_models, colour_models)

    elif args.action == 'predict':
        util.predict(cluster_split_train_df_list, cluster_val_df_list, cluster_test_df_list)

    elif args.action == 'loadpredict':
        util.load_predictions(cluster_split_train_df_list, cluster_val_df_list, cluster_test_df_list)

    util.create_submission(cluster_test_df_list)
    util.generate_meanIoU_report(cluster_split_train_df_list, cluster_val_df_list)

    
    