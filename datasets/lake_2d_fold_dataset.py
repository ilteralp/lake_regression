#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 23:03:17 2021

@author: melike
"""
import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import h5py
import os.path as osp
import sys
sys.path.append("..")
import constants as C
from datasets import Lake2dDataset

class Lake2dFoldDataset(Lake2dDataset):
    """
    Balik Lake dataset with patches as samples. Will be used with DAN. 
    
    Args:
        learning (string): Type of samples. Should be one of {'labeled', 'unlabeled'}. 
        patch_size (int): Size of the patch that is centered on the sample pixel. Should be an odd number. 
        date_type (string): Type of date label that will be used for classification. 
        Should be one of {'month', 'season', 'year'}.
        fold_setup (string): Type of fold setup. Should be one of {'spatial', 'temporal_day', 'temporal_year'}.
        ids (list of int): Either pixel or image ids depending on setup that will be used to create this dataset. 
    """
    def __init__(self, learning, date_type, fold_setup, ids, patch_size=3):
        Lake2dDataset.__init__(self, learning=learning, date_type=date_type, patch_size=patch_size)
        self.verify(fold_setup, ids)
        self.fold_setup = fold_setup.lower()
        self.ids = ids
    
    """
    Checks given params. 
    """
    def verify(self, fold_setup, ids):
        if fold_setup.lower() not in ['spatial', 'temporal_day', 'temporal_year']:
            raise Exception('fold_setup should be one of {\'spatial\', \'temporal_day\', \'temporal_year\'}')
        if fold_setup.lower() == 'spatial':
            if not np.all([i in range(0, 10) for i in ids]):
                raise Exception('Ids are expected to be in [0-9] range for the spatial setup. Given {}'.format(ids))
        if fold_setup.lower() == 'temporal_day':
            if not np.all([i in range(0, 32) for i in ids]):
                raise Exception('Ids are expected to be in [0-31] range for the temporal_day setup. Given {}'.format(ids))
        if fold_setup.lower() == 'temporal_year':
            if not np.all([i in range(0, 3) for i in ids]):
                raise Exception('Temporal_year setup supports ids [0, 1, 2] that corresponds to years [2017, 2018, 2019]. Given {}'.format(ids))
     
    """
    Returns length of the dataset. 
    """
    def __len__(self):                                                          # Does not inherit! Depends on fold_setup. 
        if self.fold_setup == 'spatial':
            num_pixels = len(self.ids) if self.learning == 'labeled' else len(self.unlabeled_mask)
            return num_pixels * len(self.img_names)                             # (10-1-1) * 32 or 75K * 32
        else:
            num_pixels = len(C.LABELED_INDICES[0]) if self.learning == 'labeled' else len(self.unlabeled_mask)
            if self.fold_setup == 'temporal_day':
                return num_pixels * len(self.ids)                               # 10 * len(ids) or 75K * len(ids)
            elif self.fold_setup == 'temporal_year':
                len_img_ids = sum([len(C.YEAR_IMG_ID[i]) for i in self.ids])    # Each year has different number of images, sum them. 
                return num_pixels * len_img_ids                                 # 10 * len(img_ids) or 75K *  len(img_ids)
    
    def __getitem__(self, index):
        pass
    
if __name__ == "__main__":
    fold_setups = ['spatial', 'temporal_day', 'temporal_year']
    learnings = ['labeled', 'unlabeled']
    for f in fold_setups:
        for l in learnings:
            dataset = Lake2dFoldDataset(learning=l, date_type='month', fold_setup=f, ids=[2, 1])
            print('learning: {}, setup: {}, len: {}'.format(l, f, len(dataset)))
            
    # dataset = Lake2dFoldDataset(learning='unlabeled', date_type='month', fold_setup='spatial', ids=[3])
    # print('len:', len(dataset))