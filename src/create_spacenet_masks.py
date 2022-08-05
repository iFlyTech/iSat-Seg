#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:56:31 2017

@author: avanetten
"""

from __future__ import print_function
import os
import sys
import time
import argparse
import pandas as pd

# add apls path and import apls_tools
# https://github.com/CosmiQ/apls/tree/master/src
path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)
import apls_tools


###############################################################################
def create_masks(path_data, buffer_meters=2, n_bands=3,
                 burnValue=150, make_plots=True, overwrite_ims=False,
                 output_df_file='',
                 header=['name', 'im_file', 'im_vis_file', 'mask_file',
                         'mask_vis_file']):
    '''
    Create masks from files in path_data.
    Write 8bit images and masks to file.
    Return a dataframe of file locations with the following columns:
        ['name', 'im_file', 'im_vis_file', 'mask_file', 'mask_vis_file']
    We record locations of im_vis_file and mask_vis_file in case im_file
      or mask_file is not 8-bit or has n_channels != [1,3]
    if using 8band data, the RGB-PanSharpen_8bit should already exist, so
 