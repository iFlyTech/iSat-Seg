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
def create_masks(path_data, 