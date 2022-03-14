
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