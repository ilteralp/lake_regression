

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:13:13 2021

@author: melike
"""

import os.path as osp

LOCAL_ENV = False
ROG_ENV = not LOCAL_ENV

""" Melike local """
if LOCAL_ENV:
    ROOT_DIR = '/home/melike/rs/balik_golu'
    MODEL_DIR_PATH = '/home/melike/repos/lake_regression/model_files'
    
""" ROG """
if ROG_ENV:
    ROOT_DIR = '/home/rog/rs/balik_golu'
    MODEL_DIR_PATH = '/home/rog/repos/lake_regression/model_files'
    
# MASK_PATH = osp.join(ROOT_DIR, 'lake_mask.png')
IMG_DIR_PATH = osp.join(ROOT_DIR, 'balik')
GT_PATH = osp.join(ROOT_DIR, 'ground_truth32.txt')
DATE_LABELS_PATH = osp.join(ROOT_DIR, 'date_labels.txt')
LABELED_INDICES = ([537, 427, 340, 263, 165, 107, 172, 249, 337, 447],
                    [235, 280, 325, 345, 398, 342, 298, 262, 225, 197])          # Or maybe vice versa ?    
IMG_SHAPE = [12, 650, 650]
SEASONS = {"'spring'" : 0, "'summer'" : 1, "'autumn'" : 2, "'winter'" : 3}
YEARS = {'2017' : 0, '2018' : 1, '2019' : 2}
DATE_TYPES = {'month' : 0, 'season' : 1, 'year' : 2}
NUM_CLASSES = {'month' : 12, 'season' : len(SEASONS), 'year' : len(YEARS)}
DAN_MODELS = ['dandadadan', 'eadan', 'eaoriginaldan']
YEAR_IMG_ID = {0: [1, 2, 3, 4, 5, 6, 7],                                  # 2017, Careful this is year-img_name mapping, not year-img_id
               1: [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], # 2018, mapping. So it's *NOT* continuous in [0, 31] range and 
               2: [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]}           # 2019, can *NOT* be used to index img_names list. 
FOLD_SETUP_NUM_FOLDS = {'spatial': 10,
                        'temporal_day': 16,
                        'temporal_year': 3,
                        'random': 10}

MLP_CFGS = {'1_hidden_layer' : [673],
            '2_hidden_layer' : [364, 256],
            '3_hidden_layer' : [292, 256, 156],
            '4_hidden_layer' : [256, 226, 192, 128],
            '5_hidden_layer' : [226, 210, 192, 156, 110],
            '6_hidden_layer' : [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
            '7_hidden_layer' : [256, 256, 256],
            '8_hidden_layer' : [192, 192, 192, 192, 192] }

""" ===================== Model params ===================== """
BATCH_SIZE = 64
# if 'lake_mask_small.png' in MASK_PATH:
#     UNLABELED_BATCH_SIZE = 512
# elif 'lake_mask_small_45.png' in MASK_PATH:
#     UNLABELED_BATCH_SIZE = 4568
# elif 'lake_mask.png' in MASK_PATH:
UNLABELED_BATCH_SIZE = 16384
BASE_LR = 0.0001