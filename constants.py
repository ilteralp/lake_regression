
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:13:13 2021

@author: melike
"""

import os.path as osp

LOCAL_ENV = True
ROG_ENV = not LOCAL_ENV

""" Melike local """
if LOCAL_ENV:
    ROOT_DIR = '/home/melike/rs/balik_golu'
    
""" ROG """
if ROG_ENV:
    ROOT_DIR = '/home/rog/rs/balik_golu'
    
MASK_PATH = osp.join(ROOT_DIR, 'lake_mask.png')
IMG_DIR_PATH = osp.join(ROOT_DIR, 'balik')
GT_PATH = osp.join(ROOT_DIR, 'ground_truth32.txt')
DATE_LABELS_PATH = osp.join(ROOT_DIR, 'date_labels.txt')
LABELED_INDICES = ([537, 427, 340, 263, 165, 107, 172, 249, 337, 447],
                    [235, 280, 325, 345, 398, 342, 298, 262, 225, 197])          # Or maybe vice versa ?    
IMG_SHAPE = [12, 650, 650]
SEASONS = {"'spring'" : 0, "'summer'" : 1, "'autumn'" : 2, "'winter'" : 3}
# LABELED_INDICES = ([537, 427], [235, 280])