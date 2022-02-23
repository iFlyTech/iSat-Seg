
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:32:20 2017

@author: avanetten

copied from https://github.com/CosmiQ/apls/tree/master/src
"""

from __future__ import print_function

import osmnx as ox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from osgeo import gdal, ogr, osr
import cv2
import subprocess
import shapely
from shapely.geometry import MultiLineString
from matplotlib.patches import PathPatch
import matplotlib.path

###############################################################################
### Previsously in apls.py
###############################################################################
def plot_metric(C, diffs, routes_str=[], 
                figsize=(10,5), scatter_png='', hist_png='',
                scatter_alpha=0.3, scatter_size=2, scatter_cmap='jet', dpi=300):
    ''' Plot outpute of cost metric in both scatterplot and histogram format'''
    
    # plot diffs
    title = 'Path Length Similarity: ' + str(np.round(C,2)) 
    fig, (ax0) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
    #ax0.plot(diffs)
    ax0.scatter(range(len(diffs)), diffs, s=scatter_size, c=diffs, 
                alpha=scatter_alpha, 
                cmap=scatter_cmap)
    if len(routes_str) > 0:
        xticklabel_pad = 0.1
        ax0.set_xticks(range(len(diffs)))
        ax0.set_xticklabels(routes_str, rotation=50, fontsize=4)
        ax0.tick_params(axis='x', which='major', pad=xticklabel_pad)

    ax0.set_ylabel('Length Diff (Normalized)')