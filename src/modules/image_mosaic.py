import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# Source: https://www.kaggle.com/bonlime/train-test-image-mosaic
def combine_images(data,indexes):
    """ Combines img from data using indexes as follows:
        0 1
        2 3 
    """
    up = np.hstack([data[indexes[0]],data[indexes[1]]])
    down = np.hstack([data[indexes[2]],data[indexes[3]]])
    full = np.vstack([up,down])
    return full

def make_mosaic(data, external_df):
    """Find images with simular borders and combine them to one big image"""
    if external_df is not None:
        external_df['mosaic_idx'] = np.nan
        external_df['mosaic_position'] = np.nan
        
    # extract borders from images
    borders = []
    for x in data:
        borders.extend([x[0,:,:].flatten(),x[-1,:,:].flatten(),
                        x[:,0,:].flatten(),x[:,-1,:].flatten()])
    borders = np.array(borders)

    # prepare df with all data
    lens = np.array([len(border) for border in borders])
    img_idx = list(range(len(data)))*4
    img_idx.sort()
    position = ['up','down','left','right']*len(data)
    nn = [None]*len(position)
    df = pd.DataFrame(data=np.vstack([img_idx,position,borders,lens,nn]).T,
                      columns=['img_idx','position','border','len','nn'])
    uniq_lens = df['len'].unique()
    
    for idx,l in enumerate(uniq_lens):
        # fit NN on borders of certain size with 1 neighbor
        nn = NearestNeighbors(n_neighbors=1).fit(np.stack(df[df.len == l]['border'].values))
        distances, neighbors = nn.kneighbors()
        real_neighbor = np.array([None]*len(neighbors))
        distances, neighbors = distances.flatten(),neighbors.flatten()

        # if many borders are close to one, we want to take only the closest
        uniq_neighbors = np.unique(neighbors)

        # difficult to understand but works :c
        for un_n in uniq_neighbors:
            # min distance for borders with same nn
            min_index = list(distances).index(distances[neighbors == un_n].min())
            # check that min is double-sided
            double_sided = distances[neighbors[min_index]] == distances[neighbors == un_n].min()
            if double_sided and distances[neighbors[min_index]] < 1000:
                real_neighbor[min_index] = neighbors[min_index]
                real_neighbor[neighbors[min_index]] = min_index
        indexes = df[df.len == l].index
        for idx2,r_n in enumerate(real_neighbor):
            if r_n is not None:
                df['nn'].iloc[indexes[idx2]] = indexes[r_n]
    
    # img connectivity graph. 
    img_connectivity = {}
    for img in df.img_idx.unique():
        slc = df[df['img_idx'] == img]
        img_nn = {}

        # get near images_id & position
        for nn_border,position in zip(slc[slc['nn'].notnull()]['nn'],
                                      slc[slc['nn'].notnull()]['position']):

            # filter obvious errors when we try to connect bottom of one image to bottom of another
            # my hypotesis is that images were simply cut, without rotation
            if position == df.iloc[nn_border]['position']:
                continue
            img_nn[position] = df.iloc[nn_border]['img_idx']
        img_connectivity[img] = img_nn

    imgs = []
    indexes = set()
    mosaic_idx = 0
    
    # errors in connectivity are filtered 
    good_img_connectivity = {}
    for k,v in img_connectivity.items():
        if v.get('down') is not None:
            if v.get('right') is not None:
                # need down right image
                # check if both right and down image are connected to the same image in the down right corner
                if (img_connectivity[v['right']].get('down') is not None) and img_connectivity[v['down']].get('right') is not None:
                    if img_connectivity[v['right']]['down'] == img_connectivity[v['down']]['right']:
                        v['down_right'] = img_connectivity[v['right']]['down']
                        temp_indexes = [k,v['right'],v['down'],v['down_right']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        # It is necessary here to filter that they are not the same
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            mosaic_idx += 1
                        continue
            if v.get('left') is not None:
                # need down left image
                if img_connectivity[v['left']].get('down') is not None and img_connectivity[v['down']].get('left') is not None:
                    if img_connectivity[v['left']]['down'] == img_connectivity[v['down']]['left']:
                        v['down_left'] = img_connectivity[v['left']]['down']
                        temp_indexes = [v['left'],k,v['down_left'],v['down']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            mosaic_idx += 1 
                            
                        continue
        if v.get('up') is not None:
            if v.get('right') is not None:
                # need up right image
                if img_connectivity[v['right']].get('up') is not None and img_connectivity[v['up']].get('right') is not None:
                    if img_connectivity[v['right']]['up'] == img_connectivity[v['up']]['right']:
                        v['up_right'] = img_connectivity[v['right']]['up']
                        temp_indexes = [v['up'],v['up_right'],k,v['right']]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            
                            mosaic_idx += 1 
                        continue
            if v.get('left') is not None:
                # need up left image
                if img_connectivity[v['left']].get('up') is not None and img_connectivity[v['up']].get('left') is not None:
                    if img_connectivity[v['left']]['up'] == img_connectivity[v['up']]['left']:
                        v['up_left'] = img_connectivity[v['left']]['up']
                        temp_indexes = [v['up_left'],v['up'],v['left'],k]
                        if (len(np.unique(temp_indexes)) < 4) or (len(indexes.intersection(temp_indexes)) > 0):
                            continue
                        good_img_connectivity[k] = temp_indexes
                        indexes.update(temp_indexes)
                        imgs.append(combine_images(data,temp_indexes))
                        
                        if external_df is not None:
                            external_df['mosaic_idx'].iloc[temp_indexes] = mosaic_idx
                            external_df['mosaic_position'].iloc[temp_indexes] = ['up_left','up_right','down_left','down_right']
                            
                            mosaic_idx += 1 
                        continue

    
    # list of not combined images. return if you need
    not_combined = list(set(range(len(data))) - indexes)
    #print('Image indexes that could not be combined: ' + str(not_combined))
    if len(not_combined) + len(imgs)*4 != len(data):
        print('WARNING! the original number of samples is: ' + str(len(data)) + 
              'the new number of samples is:' + str(len(not_combined) + len(imgs)*4))
    
    if external_df is not None:
        external_df.loc[external_df[external_df['mosaic_idx'].isnull()].index,'mosaic_idx'] = range(
            int(np.nanmax(external_df.mosaic_idx.unique())) + 1,
            int(np.nanmax(external_df.mosaic_idx.unique())) + 1 + len(external_df.mosaic_idx[external_df.mosaic_idx.isnull()]))
        external_df['mosaic_idx'] = external_df['mosaic_idx'].astype(np.int32)
        
    return imgs, external_df, good_img_connectivity, not_combined


def split_mosaic_image(image):
    """ Splits a mosaic image generated from combine_images, using indexes as follows:
        0 1
        2 3 
    """
    split_images = []
    height = int(image.shape[0] / 2)
    width = int(image.shape[1] / 2)
    split_images.append(image[0:height,0:width])
    split_images.append(image[0:height,width:])
    split_images.append(image[height:,0:width])
    split_images.append(image[height:,width:])
    return split_images


def get_mosaic_image_part(image, mosaic_ix):
    """ Splits a mosaic image generated from combine_images, using indexes as follows:
        0 1
        2 3 
        Only the image portion at the specified index is returned.
    """
    split_image_part = []
    height = int(image.shape[0] / 2)
    width = int(image.shape[1] / 2)
    if mosaic_ix == 0:
        split_image_part = image[0:height,0:width]
    elif mosaic_ix == 1:
        split_image_part = image[0:height,width:]
    elif mosaic_ix == 2:
        split_image_part = image[height:,0:width]
    else:
        split_image_part = image[height:,width:]

    return split_image_part


def merge_mosaic_images(mosaic_dict, mosaic_images, orig_images, Y_orig=None):
    """ Merge the list of mosaic images with all original images.

    Args:
        mosaic_dict: Dictionary specifying how mosaic images were created, returned from make_mosaic
        mosaic_images: List of all mosaic images returned from make_mosaic
        orig_images: List of all images, some (or all, or none) of which were used to generate the mosaic images
        Y_orig: If building mosaic images for training, the Y/expected images corresponding to orig_images

    Returns:
        3 lists - merged_images, merged_sizes, merged_Y (empty list if Y_orig was not provided). This list of
        images can then be resized, windowed, etc., and provided as input images for training or predictions.
        To split the merged list back into the separate portions, use split_merged_mosaic.
    """
    orig_index = list(range(0, len(orig_images)))
    merged_images = []
    merged_sizes = []
    merged_Y = []

    # If Y/expected values are desired, construct the merged Y
    # images to correspond with the mosaic images.
    if Y_orig:
        for k, v in mosaic_dict.items():
            merged_Y.append(combine_images(Y_orig, v))

    # Mosaic images are output first
    for img in mosaic_images:
        merged_images.append(img)
        merged_sizes.append([img.shape[0], img.shape[1]])

    mosaic_all_ix=[]
    [mosaic_all_ix.extend(v) for v in mosaic_dict.values()]
    leftovers = [x for x in orig_index if x not in mosaic_all_ix]

    # And then output all images that are not part of a larger mosaic image
    for ix in leftovers:
        leftover_img = orig_images[ix]
        merged_images.append(leftover_img)
        merged_sizes.append([leftover_img.shape[0], leftover_img.shape[1]])
        if Y_orig:
            merged_Y.append(Y_orig[ix])

    return (merged_images, merged_sizes, merged_Y)


def split_merged_mosaic(mosaic_dict, merged_images, num_orig_images):
    """ Splits the merged mosaic images list from merge_mosaic_images back into the original image parts.

    Args:
        mosaic_dict: Dictionary specifying how mosaic images were created, returned from make_mosaic
        merged_images: List of all merged images returned from merge_mosaic_images. This could also be the
            the list of images predicted from those merged images. The width and height of each merged_image must
            match the original merged_images list, however the number of channels is allowed to be different.
        num_orig_images: The number of original images in the orig_images list provided to merge_mosaic_images.

    Returns:
        The original list in the same order as the orig_images passed into merge_mosaic_images.
    """
    orig_index = list(range(0, num_orig_images))
    mosaic_all_ix=[]
    [mosaic_all_ix.extend(v) for v in mosaic_dict.values()]
    list.sort(mosaic_all_ix)
    leftovers = [x for x in orig_index if x not in mosaic_all_ix]

    split_images = []

    num_mosaic_images = len(mosaic_dict)
    leftover_ix = num_mosaic_images

    for ix in range(0, num_orig_images):
        if ix in leftovers:
            split_images.append(merged_images[leftover_ix])
            leftover_ix += 1
        else:
            # Need to find which mosaic image has the part we want
            mosaic_image_num = 0
            for mosaic_parts in mosaic_dict.values():
                if ix in mosaic_parts:
                    mosaic_image = merged_images[mosaic_image_num]
                    mosaic_part_ix = mosaic_parts.index(ix)
                    split_images.append(get_mosaic_image_part(mosaic_image, mosaic_part_ix))
                    break
                mosaic_image_num += 1
                
    actual_len = len(split_images)
    expected_len = num_orig_images
    assert actual_len == expected_len, 'ERROR: Unable to recombine all mosaic images, expected: ' + str(expected_len) + ', actual: ' + str(actual_len)
    
    return split_images