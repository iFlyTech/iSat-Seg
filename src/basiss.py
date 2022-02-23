
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