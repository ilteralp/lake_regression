#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 22:29:40 2021

@author: melike
"""
from torch.utils.data import Dataset
import numpy as np
import os
import sys
import csv
import os.path as osp
from skimage import io
sys.path.append("..")
import constants as C

class BaseLakeDataset(Dataset):
    """
    Base class for Balik Lake dataset. 
    
    Args:
        learning (string): Type of samples. Should be one of {'labeled', 'unlabeled'}. 
        date_type (Optional, string): Type of date label that will be used for classification. 
        Should be one of {'month', 'season', 'year'}.
    """
    def __init__(self, learning, date_type, patch_size):
        if not self._verify(learning=learning, date_type=date_type, patch_size=patch_size):
            raise Exception('Params should be one from each, learning: {labeled, unlabeled} and date_type: {month, season, year}.')
        self.learning = learning.lower()
        self.date_type = None if date_type is None else date_type.lower()
        self.patch_size = patch_size
        self.img_names = self._get_image_names()
        self.dates = self._read_date_info()
        self.patch_means = None
        self.patch_stds = None
        self.reg_mean = None
        self.reg_std = None
        self.reg_min = None
        self.reg_max = None
        
        
        if self.learning == 'labeled':                                          # Load regression values for labeled samples
            self.reg_vals = self._read_GT()
        else:
            self.unlabeled_mask = self._load_unlabeled_mask()                   # Load unlabeled mask where labeled samples are removed. 
            
    def _verify(self, learning, date_type, patch_size):
        return learning.lower() in ['labeled', 'unlabeled'] and date_type.lower() in ['month', 'season', 'year'] and patch_size in [3, 5, 7, 9]
            
    def __len__(self):                                                          # unlabeled: 32 * 75662 = 2421184, labeled: 32 * 10
        return 32 * len(self.unlabeled_mask) if self.learning == 'unlabeled' else len(self.reg_vals) 
        # Check len of splits in case of labeled samples !
        
    """
    Returns image names where empty folders are skipped. 
    """
    def _get_image_names(self):
        ls = os.listdir(C.IMG_DIR_PATH)
        num_names = sorted([int(l) for l in ls if l not in['22', '23']])        # Skip empty folders
        return [str(l) for l in num_names]
    
    """
    Reads ground-truth file and keeps regression value of labeled samples. 
    """
    def _read_GT(self):
        if self.learning != 'labeled':
            raise Exception('Only labeled samples have regression values!')
        with open(C.GT_PATH) as f:
            reader = csv.reader(f, delimiter='\t')
            return np.array([float(row[2]) for row in reader])
    
    """
    Returns regression value of for given indices. Only for labeled samples. 
    Each measured lake sample has 32 values in GT. 
    """
    def _get_regression_val(self, img_idx, px_idx):
        ln = 32 * px_idx + img_idx
        if(len(self.reg_vals) < ln):
            raise Exception('Accessing index#{} where total length is {} at {}'.format(ln, len(self.reg_vals), C.GT_PATH))
        return self.reg_vals[ln]
    
    """
    Reads month, season and year info of all images. 
    """
    def _read_date_info(self):
        with open(C.DATE_LABELS_PATH, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            return [{'month' : int(l[1]) - 1, 'season' : C.SEASONS[l[2]],       # Months start from 1. Converts seasons to {0, 1, 2, 3}
                     'year' : C.YEARS[l[3]]} for l in reader]                   # Converts years to {0, 1, 2}. 
                     # 'year' : int(l[3])} for l in reader]
    """
    Loads lake mask of unlabeled samples. Within it, labeled samples are also false. 
    """
    def _load_unlabeled_mask(self):
        # img = io.imread(C.MASK_PATH)
        img = io.imread(osp.join(C.ROOT_DIR, 'eroded_bin_lake_mask_{}.png'.format(self.patch_size)))
        img = img[:,:,:3] if img.shape[2] == 4 else img                         # Check alpha channel    
        mask = np.all(img == (255, 0, 255), axis=-1)                            # Lake mask, 650x650
        # print('before', mask[C.LABELED_INDICES])
        # print('before, sum of mask pixels:', np.sum(mask))
        mask[C.LABELED_INDICES] = False                                         # Set labeled indices to false.  
        self.num_samples = np.sum(mask)
        # print('after removing labeled samples, number of unlabeled samples:', self.num_samples)
        return mask
    
    """
    Sets patch means and stds of dataset.
    """
    def set_patch_mean_std(self, means, stds):
        self.patch_means = means
        self.patch_stds = stds
        
    """
    Sets regression values' mean and std of dataset. 
    """
    def set_reg_mean_std(self, reg_mean, reg_std):
        self.reg_mean = reg_mean
        self.reg_std = reg_std
    
    """
    Sets min and max of regression values. 
    """
    def set_reg_min_max(self, reg_min, reg_max):
        self.reg_min = reg_min
        self.reg_max = reg_max
    
