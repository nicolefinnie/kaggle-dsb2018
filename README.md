# In a nutshell
Also see the post here 
https://www.kaggle.com/c/data-science-bowl-2018/discussion/54742

**Newbies' solution - as first time Kagglers (LB = 0.545)**

I'm @CPMP's team mate + colleague.  Since most of us were first time Kagglers, we implemented a very simple U-net, taken from the U-net starter of  @Kjetil Åmdal-Sævik that any DL newbies like us can do.

**Model**

- Same as the U-net starter except two output channels: `masks` and `contours` with the `binary cross entropy loss`

**Pre-processing**
 
   - CLAHE (Contrast Limited Adaptive Histogram Equalization) 
   - L-channel only
   - Inversion only when the background is lighter than the foreground
   - Reference: https://www.kaggle.com/kmader/normalizing-brightfield-stained-and-fluorescence
   - The goal is to make it easy for the U-net to learn their common features.


**K-mean cluster based on colours**

  -  K-mean cluster using `R, G, R-G deviation` on train/test images. As @Heng suggested, if we train grey and colour images together, their common trait is the shape/boundary, so it'd be hard for the neural net to learn the features inside the nucleus. In a result, our U-net did fairly well on the colour images at the 2nd stage, okay, maybe not too well on @Allen's art :)

![Allen's art](https://kaggle2.blob.core.windows.net/forum-message-attachments/315594/9206/AllensArt.png)

**Mosaic for training and predicting**

- As suggested by @CPMP, we applied @Emil's mosaic idea (KNN on 4 edges of the image to find the matching neighbours) and merged all images to mosaic to gain the information on the edges 
https://www.kaggle.com/bonlime/train-test-image-mosaic
![A mosaic trained image composed of 4 images](https://kaggle2.blob.core.windows.net/forum-message-attachments/315594/9207/mosaic_train_colour1.png)

- Around `37%` of the grey test images and about `77%` of the colour test images can be formed into mosaic, this ratio still holds true for the stage 2 test images. That means we gained more hidden traits than most teams.

**Windowing images for training and predictions**

- Thanks to mosaic, now we have a lot more information on the edges, so we crop images to `256x256` using overlapping windows with a stride of `128`, because `256x256` yields the best mean IoU.

- Mirror the edges and the corners as suggested by the U-net paper. However, we have less edges than the people who don't use the `mosaic` approach

- Rotate and flip all windowing images, this augments images by `8x`

**Data augmentation**
 
- Transform perspective using the `imgaug` library. (this approach is used by the top 1 team)

- Additive Gaussian noise + speckle noise. (used by the top 1 team)

- Gaussian Blur (no time to train)

- Image pyramid, upsampling and downsampling based on the average nuclei size per image (no time to train )

- Regional gamma correction (no time to train) 

- Greyscale (no time to train)

- Inversion, hurts our LB since it makes the U-net difficult to learn, so we discarded the idea just like the top 1 team

**Average windowing predictions**

- Predict on overlapping windowing images in rotated/flipped/mirrored versions
- Average all windowing predictions to minimize the artifacts, this approach was used by the winners in the satellite competition last year.
- Stitch predictions together to the original size


**Post-processing**

- Otsu threshold on predicted masks and predicted contours
- Find sure foreground: `(predicted masks - predicted contours)` 
- Find sure background: dilates the predicted masks
- Find unknown area: `(sure background - sure foreground)`
- Find labels on the sure foreground 
- Random walker or watershed on the masks + contours to avoid `contour erosion`, adding the thresholding contour gave us a big LB boost. 

- Reference: https://docs.opencv.org/3.3.1/d3/db4/tutorial_py_watershed.html

**Model ensemble using weighted majority voting**

- @CPMP chose weights based on our public LB scores
- `4 * ( baseline model + noise ) + 2 * ( baseline model + transform ) + 1 * ( baseline model)`


**Acknowledgement**

Thanks to @CPMP. If he didn't join our team in the last 3 days, we wouldn't have known that we had to train and submit models by the deadline of the first stage and we wouldn't have submitted any models that we could have used at the 2nd stage. Sadly, we didn't get to train all the models with data augmentation we planned to, however, we're happy with the results as first time Kagglers. It's pretty easy to implement our approach. For our uploaded instruction for the first stage, see below. 


<br>


# Schwäbische Nuclei U-Net 

## Definition of our training data
- [`stage1_train_fixed.zip `](https://drive.google.com/open?id=1tVUQjHYZqyAZ7QeIAWQ_FlP1xReSZ0Kv) with `md5sum 9a3e938a312baa30fcea84c476a278cb` 
We merged the original `stage1_train.zip` with [this fixed stage1 train data](https://github.com/lopuhin/kaggle-dsbowl-2018-dataset-fixes/tree/master/stage1_train),
generated the combined masks and contours under `prep_masks` under each imageID directory, and programmatically removed the following imageIDs from the training set on the fly.
* ###### 7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80
* ###### adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df
* ###### 12aeefb1b522b283819b12e4cfaf6b13c1264c0aadac3412b4edd2ace304cb40
- For stage 2, if allowed, we'll add [`stage1_test.zip`](https://www.kaggle.com/c/data-science-bowl-2018/data) with the released masks to our training data

## How to reproduce our submissions


### Train our colour models

- Train 3 models with full size of colour train images with data augmentation and then window on all augmented data and rotate/flip on all windowing data for 10 epochs. With a limitation of total training image size of roughly 300k on AWS with one GPU V100


- model 1c mosaic with Gaussian noise and speckle noise data augmentation, `sigma = 0.1`

`python schwaebische_nuclei.py train --maxtrainsize 300000 --mosaic --noise 0.1 --rotate --epoch 10 --valsplit 0 --colouronly`

- model 2c mosaic with perspective transform data augmentation, `sigma = 0.175`

`python schwaebische_nuclei.py train --maxtrainsize 300000 --mosaic --transform 0.175 --rotate --epoch 10 --valsplit 0 --colouronly`

- model 3c mosaic only 

`python schwaebische_nuclei.py train --maxtrainsize 300000 --mosaic --rotate --epoch 10 --valsplit 0 --colouronly`



### Train our grey models and retrain our colour models

- Train 3 models with full size of grey/colour train images with data augmentation and then window on all augmented data and rotate/flip on all windowing data for 10 epochs. With a limitation of total training image size of roughly 300k on AWS with one GPU V100

- model 1g mosaic with Gaussian noise and speckle noise data augmentation, `sigma = 0.1`

`python schwaebische_nuclei.py train --maxtrainsize 300000 --mosaic --noise 0.1 --rotate --epoch 10 --valsplit 0 --loadmodel output_from_model_1c/`


- model 2g mosaic with perspective transform data augmentation, `sigma = 0.175`

`python schwaebische_nuclei.py train --maxtrainsize 300000 --mosaic --transform 0.175 --rotate --epoch 10 --valsplit 0 --loadmodel output_from_model_2c/`


- model 3g mosaic only 

`python schwaebische_nuclei.py train --maxtrainsize 300000 --mosaic --rotate --epoch 10 --valsplit 0 --loadmodel output_from_model_3c/`


### Model predictions blending 

#### Predict on grey images using each grey models: `model_1g, model_2g, model_3g`

- Window and rotate/flip test images and predict on those grey test images using each model

- Average weighted predictions of `model_1g`, `model_2g` and `model_3g` with weight distribution `[4, 2, 1]`:

`predictions = ( 4 * model_1g + 2 * model_2g + 1 * model_1g) / 7`

- Predictions on contours and masks will be generated automatically and saved to the disk

#### Predict on colour images using each colour models: `model_1c, model_2c, model_3c`

- Window and rotate/flip test images and predict on those colour test images using each model

- Average weighted predictions of `model_1c`, `model_2c` and `model_3c` with weight distribution `[4, 2, 1]`:

`predictions = ( 4 * model_1c + 2 * model_2c + 1 * model_3c ) / 7` by issuing the following command:


```
python schwaebische_nuclei.py predict --loadmodel output_from_all_models/ --weights 4 2 1
```

### Generate submissions

- After weighted predictions get averaged, CSV will be generated by the above command, we use this for our submissions
- Predictions on contours and masks will be generated automatically and saved to the disk
 



