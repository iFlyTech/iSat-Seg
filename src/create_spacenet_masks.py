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
        3band should be run prior to 8band
    '''
    
    t0 = time.time()
    # set paths
    #path_apls = os.path.dirname(path_apls_src)
    #path_data = os.path.join(path_apls, args.test_data_loc)
    path_labels = os.path.join(path_data, 'geojson/spacenetroads')
    # output directories
    path_masks = os.path.join(path_data, 'masks_' + str(buffer_meters) + 'm')
    path_masks_plot = os.path.join(path_data, 'masks_' \
                                   + str(buffer_meters) + 'm_plots')
    # image directories
    path_images_vis = os.path.join(path_data, 'RGB-PanSharpen_8bit')
    if n_bands == 3:
        path_images_raw = os.path.join(path_data, 'RGB-PanSharpen')
        path_images_8bit = os.path.join(path_data, 'RGB-PanSharpen_8bit')
    else:
        path_images_raw = os.path.join(path_data, 'MUL-PanSharpen')
        path_images_8bit = os.path.join(path_data, 'MUL-PanSharpen_8bit')
        if not os.path.exists(path_images_vis):
            print ("Need to run 3band prior to 8band!")
            return
        
    # create directories
    for d in [path_images_8bit, path_masks, path_masks_plot]:
        if not os.path.exists(d):
            os.mkdir(d)
    
    # iterate through images, convert to 8-bit, and create masks
    outfile_list = []
    im_files = os.listdir(path_images_raw)
    nfiles = len(im_files)
    for i,im_name in enumerate(im_files):
        if not im_name.endswith('.tif'):
            continue
                    
        # define files
        name_root = 'AOI' + im_name.split('AOI')[1].split('.')[0]
        im_file_raw = os.path.join(path_images_raw, im_name)
        im_file_out = os.path.join(path_images_8bit, im_name)
        im_file_out_vis = im_file_out.replace('MUL', 'RGB')
        ## get visible file (if using 8band imagery we want the 3band file
        ##   for plotting purposes)
        #if n_bands == 3:
        #    im_file_out_vis = im_file_o