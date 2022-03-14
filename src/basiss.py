
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:26:51 2018

@author: avanetten
"""

from __future__ import print_function

import numpy as np
import argparse
import time
import os
import datetime
import gdal
import pandas as pd
import cv2
#import json
#import tensorflow as tf

from keras.models import Model
from keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, 
                          Lambda, Convolution2D, Conv2D, MaxPooling2D, 
                          UpSampling2D, Input, merge, Concatenate, 
                          concatenate, Conv2DTranspose)
from keras.callbacks import ModelCheckpoint, EarlyStopping
                        #,LearningRateScheduler
from keras.optimizers import SGD, Adam, Adagrad
from keras import backend as K
import keras.utils.vis_utils

#from keras.regularizers import l2
#from keras.utils.np_utils import to_categorical
#from keras.initializers import Identity
#from keras.layers.core import Permute
#from keras.layers.normalization import BatchNormalization


###############################################################################
### Jaccard
###############################################################################
def jaccard_coef(y_true, y_pred, smooth=1e-12):
    '''https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    # __author__ = Vladimir Iglovikov'''
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)

###############################################################################
def jaccard_coef_int(y_true, y_pred, smooth=1e-12):
    '''https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
    # __author__ = Vladimir Iglovikov'''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)

###############################################################################
###############################################################################
### Manually define metrics
# define metrics from https://github.com/fchollet/keras/blob/master/keras/metrics.py
#   since for some reason keras can't always import them
###############################################################################    
def dice_coeff(y_true, y_pred):
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersect = K.sum(y_true_flat * y_pred_flat)
    return (2. * intersect) / (K.sum(y_true_flat) + K.sum(y_pred_flat))

###############################################################################
def dice_loss(y_true, y_pred):
    return -1. * dice_coeff(y_true, y_pred)
    
###############################################################################
def mse(y_true, y_pred):
    return K.mean(K.square(K.flatten(y_pred) - K.flatten(y_true)), axis=-1)

###############################################################################
def f1_score(y_true, y_pred):
    '''https://stackoverflow.com/questions/45411902/how-to-use-f1-score-with-keras-model'''

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score 

###############################################################################
def f1_loss(y_true, y_pred):
    return 1. - f1_score(y_true, y_pred)


###############################################################################
### Data load functions
###############################################################################
def load_multiband_im(image_loc, nbands=3):
    '''use gdal to laod multiband files, else use cv2'''
    
    if nbands == 1:
        img = cv2.imread(image_loc, 0)
    elif nbands == 3:
        img = cv2.imread(image_loc, 1)
    else:
        #ingest 8 band image
        im8_raw = gdal.Open(image_loc)
        bandlist = []
        for band in range(1, im8_raw.RasterCount+1):
            srcband = im8_raw.GetRasterBand(band)
            band_arr_tmp = srcband.ReadAsArray()
            bandlist.append(band_arr_tmp)
        img = np.stack(bandlist, axis=2)
    
    return img


###############################################################################    
def load_files_universal(file_list_loc, nbands=3, n_classes=2,
                         output_size=(), max_len=10000000,
                         export_im_vis_arr=False, 
                         reshape_mask=False, super_verbose=False):
    '''Load image and mask files from file list
    any number of bands is acceptable
    file_list_loc has rows:
        [im_test_root, im_test_file, im_vis_file, mask_file, mask_vis_file])
    '''

    print ("\nLoading files from ", file_list_loc, "...")
    print ("  output_size:", output_size)

    df = pd.read_csv(file_list_loc, na_values='')
    
    #########################
    N = len(df)
    if len(output_size) > 0:
        h, w = output_size[0], output_size[1]
    else:
        # read in first row to get shape
        [im_test_root, im_test_file, im_vis_file, mask_file, \
                                             mask_vis_file] = df.iloc[0]
        sat = load_multiband_im(im_test_file, nbands=nbands)
        h, w, _ = sat.shape
    # set shapes
    im_shape = (N, h, w, nbands)
    mask_shape = (N, h, w, n_classes)
    #########################    

    # turning a list to an array often saturates memory, so create empty
    #   np arrays first
    im_arr = np.empty(im_shape, dtype=np.uint8)
    im_vis_arr = np.empty(im_shape, dtype=np.uint8)
    mask_arr = np.empty(mask_shape, dtype=np.uint8)
    name_arr = []
    for i,row in enumerate(df.values):
        
        if i > max_len:
            break
        
        [im_test_root, im_test_file, im_vis_file, mask_file, \
                                                  mask_vis_file] = row
        # load image
        sat = load_multiband_im(im_test_file, nbands=nbands)
        
        if (i % 50) == 0:
            print ("Loading file", i, "of", len(df))
            print ("   File path:", im_test_file)
            print ("   im.shape:", sat.shape)
            print ("   im.dtype:", sat.dtype)
            
        # usually, we'll skip im_vis_file
        if export_im_vis_arr:
            sat_vis = load_multiband_im(im_vis_file, nbands=nbands)
        else:
            sat_vis = []

        if n_classes <= 2:
            # if file name is null, set to ''
            #if np.isnan(mask_file):
            #    mask_file = ''
            # create mask
            if len(mask_file) > 0:
                # assume mask is has 1 layer!
                mask = load_multiband_im(mask_file, 1)  
            else:
                mask = np.zeros((sat.shape[0], sat.shape[1]))
        else:
            print ("Need write some more code for n_classes > 2!")
            break

        ################################
        # resize files, if desired
        if super_verbose:
            print ("mask.shape:", mask.shape)

        # set image size
        if len(output_size) > 0 :
            resize_size = output_size
        else:
            h, w = mask.shape[:2]
            resize_size = (h,w)
            
        # resize
        im_resize = cv2.resize(sat, resize_size)
        
        if export_im_vis_arr:
            im_vis_resize = cv2.resize(sat_vis, resize_size)
        else:
            im_vis_resize = []

        # make mask of appropriate dept (include background channel)
        if n_classes == 2:
            roads_resize = cv2.resize(mask, resize_size)
            bg_resize = np.array(np.ones(resize_size) \
                                 - roads_resize).astype(int)
            mask_resize = np.stack((bg_resize, roads_resize), axis=2)
            #mask_vis_resize = np.stack((bg_resize, roads_resize), axis=2)
        elif n_classes == 1:
            mask_resize = cv2.resize(mask, resize_size)
        else:
            print ("Need write some more code for n_classes > 2!")
            break
        #print "mask_resize.shape:", mask_resize
        ################################

        ################################
        # add to arrays
        name_arr.append(im_test_root)
        mask_arr[i] = mask_resize
        im_arr[i] = im_resize
        if export_im_vis_arr:
            im_vis_arr[i] = im_vis_resize
        # delete to save memory
        del row, sat, sat_vis, mask, im_resize, mask_resize

    # reshape mask, if desired
    if reshape_mask or n_classes==1:
        mask_arr0 = np.array(mask_arr)
        #print "mask_arr0.shape:", mask_arr0.shape
        N,h,w = mask_arr0.shape
        mask_arr = np.reshape(mask_arr0, (N,h,w,1))
    
    # reshape im_vis if only 1 band, since keras needs a tensor (N,h,w,nbands)
    if nbands == 1:
        tmp_arr = np.array(im_arr)
        N,h,w = tmp_arr.shape
        if super_verbose:
            print ("im_arr.shape:", tmp_arr.shape)
        im_arr = np.reshape(np.array(im_arr),  (N,h,w,1))
        im_vis_arr = np.reshape(np.array(im_vis_arr),  (N,h,w,1))
    
    
    print ("create np name arr..")
    name_arr = np.asarray(name_arr)
    print ("name_arr.shape:", name_arr.shape)

    return name_arr, im_arr, im_vis_arr, mask_arr


###############################################################################
def slice_ims(im_arr, mask_arr, names_arr, slice_x, slice_y, 
                    stride_x, stride_y,
                    pos_columns = ['idx', 'name', 'xmin', 'ymin', 'slice_x', 
                                   'slice_y', 'im_x', 'im_y'],
                    verbose=True, super_verbose=False):
    '''Slice images into patches, assume ground truth masks are present'''
    
    if verbose:
        print ("Slicing images and masks...")
    t0 = time.time()    
    mask_buffer = 0
    count = 0
    im_list, mask_list, name_list, pos_list = [], [], [], []
    nims,h,w,nbands = im_arr.shape
    for i, (im, mask, name) in enumerate(zip(im_arr, mask_arr, names_arr)):

        seen_coords = set()
        
        if verbose and (i % 100) == 0:
            print (i, "im_shape:", im.shape, "mask_shape:", mask.shape)
                
        # dice it up
        # after resize, iterate through image and bin it up appropriately
        for x in range(0, w - 1, stride_x):  
            for y in range(0, h - 1, stride_y): 
                
                xmin = min(x, w-slice_x)
                ymin = min(y, h - slice_y) 
                coords = (xmin, ymin)
                
                # check if we've already seen these coords
                if coords in seen_coords:
                    if super_verbose:
                        print ("coords already encountered ", \
                               "(xmin, ymin):", coords)
                    continue
                else:
                    seen_coords.add(coords)
                
                # check if we screwed up binning
                if (xmin + slice_x > w) or (ymin + slice_y > h):
                    print ("Improperly binned image, see slice_ims()")
                    return

                # get satellite image cutout
                im_cutout = im[ymin:ymin + slice_y, xmin:xmin + slice_x]
                
                ##############
                # skip if the whole thing is black
                if np.max(im_cutout) < 1.:
                    if super_verbose:
                        print ("Cutout null, skipping...")
                    continue
                else:
                    count += 1
                ###############
                
                # get mask cutout
                x1, y1 = xmin + mask_buffer, ymin + mask_buffer
                mask_cutout  = mask[y1:y1 + slice_y, x1:x1 + slice_x]
                
                if super_verbose:
                    print ("x, y:", x, y )
                    print ("  x_min, y_min:", xmin, ymin)
                    print ("  x_bounds:", xmin, xmin+slice_x, 
                                   "y_bounds:", ymin, ymin+slice_y)
                    print ("  im_cutout.shape:", im_cutout.shape)
                    print ("  mask_cutout.shape:", mask_cutout.shape)

                # set slice name
                name_full = str(i) + '_' + name + '_' \
                    + str(xmin) + '_' + str(ymin) + '_' + str(slice_x) \
                    + '_' + str(slice_y)  + '_' + str(w) + '_' + str(h)
                pos = [i, name, xmin, ymin, slice_x, slice_y, w, h]
                # add to arrays
                #idx_list.append(idx_full)
                name_list.append(name_full)
                im_list.append(im_cutout)
                mask_list.append(mask_cutout) 
                pos_list.append(pos)
    
    # convert to np arrays
    del im_arr
    del mask_arr
    #idx_out_arr = np.array(idx_list)
    name_out_arr = np.array(name_list)
    im_out_arr = np.array(im_list)
    mask_out_arr = np.array(mask_list)
    
    # create position datataframe
    df_pos = pd.DataFrame(pos_list, columns=pos_columns)
    df_pos.index = np.arange(len(df_pos))
    
    if verbose:
        print ("  im_out_arr.shape;", im_out_arr.shape)
        print ("  mask_out_arr.shape:", mask_out_arr.shape)
        print ("  mask_out_arr[0] == mask_out_arr[1]?:", 
                   np.array_equal(mask_out_arr[0], mask_out_arr[1]))
        print ("  time to slice arrays:", time.time() - t0, "seconds")
        
    return df_pos, name_out_arr, im_out_arr, mask_out_arr


###############################################################################
### Define model(s)
###############################################################################
def unet(input_shape, n_classes=2, kernel=3, loss='binary_crossentropy', 
         optimizer='SGD', data_format='channels_last'):
    '''
    https://arxiv.org/abs/1505.04597
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    '''
    
    print ("UNET input shape:", input_shape) 
    #inputs = Input((input_size, input_size, n_channels))
    inputs = Input(input_shape)
    
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    #up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same')(conv6)
