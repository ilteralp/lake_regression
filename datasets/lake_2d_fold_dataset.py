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
import itertools
import sys
sys.path.append("..")
import constants as C
from datasets import Lake2dDataset

class Lake2dFoldDataset(Lake2dDataset):
    """
    Balik Lake dataset with patches as samples. Will be used with DAN. 
    
    Args:
        learning (string): Type of samples. Should be one of {'labeled', 'unlabeled'}. 
        date_type (string): Type of date label that will be used for classification. 
        Should be one of {'month', 'season', 'year'}.
        fold_setup (string): Type of fold setup. Should be one of {'spatial', 'temporal_day', 'temporal_year'}.
        ids (list of int, optional): Either pixel or image ids depending on setup that will be used to create this dataset. 
        Ids are None in case of fold_setup='spatial' and learning='unlabeled' since there are 75K pixels per image that 
        cannot be indexed. For fold_setup='temporal_day' and fold_setup='temporal_year', ids [22, 23] should not be used. 
        patch_size (int, optional): Size of the patch that is centered on the sample pixel. Should be an odd number. 
    """
    def __init__(self, learning, date_type, fold_setup, ids=None, patch_size=3):
        Lake2dDataset.__init__(self, learning=learning, date_type=date_type, patch_size=patch_size)
        self.verify(fold_setup, ids)
        self.fold_setup = fold_setup.lower()
        self.ids = ids
        self.img_ids = self._init_image_ids()
        self.glob_img_id_dict = self._init_global_image_id_dict()
    
    """
    Checks given params. 
    """
    def verify(self, fold_setup, ids):
        if fold_setup.lower() not in ['spatial', 'temporal_day', 'temporal_year']:
            raise Exception('fold_setup should be one of {\'spatial\', \'temporal_day\', \'temporal_year\'}')
        if fold_setup.lower() == 'spatial':
            if self.learning == 'labeled':
                if len(ids) > 10:
                    raise Exception('Expected at most 10 ids for spatial setup. Given {} ids.'.format(len(ids)))
                if not np.all([i in range(0, 10) for i in ids]):
                    raise Exception('Ids are expected to be in [0-9] range for the spatial setup. Given {}'.format(ids))
            else:
                if ids is not None:
                    raise Exception('Expected ids to be None in case of fold_setup=\'{}\' and learning=\'{}\'.'.format(fold_setup, self.learning))
        if fold_setup.lower() == 'temporal_day':
            if len(ids) > 32:
                raise Exception('Expected at most 32 ids for temporal_day setup. Given {} ids.'.format(len(ids)))
            if ids in [22, 23]:
                raise Exception('Image ids [22, 23] are invalid.')
            if not np.all([i in range(0, 32) for i in ids]):
                raise Exception('Ids are expected to be in [0-31] range for the temporal_day setup. Given {}'.format(ids))
        if fold_setup.lower() == 'temporal_year':
            if len(ids) > 3:
                raise Exception('Expected at most 3 ids for temporal_day setup. Given {} ids.'.format(len(ids)))     
            if not np.all([i in range(0, 3) for i in ids]):
                raise Exception('Temporal_year setup supports ids [0, 1, 2] that corresponds to years [2017, 2018, 2019]. Given {}'.format(ids))
    """
    Returns ids of images. Image names are not kept due to accessing it with global indices. 
    """
    def _init_image_ids(self):
        if self.fold_setup == 'spatial':                                        # Image ids are [0, 31]
            return [*range(len(self.img_names))]
        elif self.fold_setup == 'temporal_day':                                 # Image ids correspond to ids for temporal_day. 
            return self.ids
        elif self.fold_setup == 'temporal_year':
            return [n for i in self.ids for n in C.YEAR_IMG_ID[i]]              # Concat list of image ids for each year id. 
                
    """
    Takes an image id, returns its index in self.img_names which must be in [0, 31]. 
    """
    def _init_global_image_id_dict(self):
        return {int(n) : k for k, n in enumerate(self.img_names)}
     
    """
    Returns length of the dataset. 
    """
    def __len__(self):                                                          # Does not inherit! Depends on fold_setup. 
        if self.fold_setup == 'spatial':
            num_pixels = len(self.ids) if self.learning == 'labeled' else len(self.unlabeled_mask)             # (10-1-1) * 32 or 75K * 32
        else:
            num_pixels = len(C.LABELED_INDICES[0]) if self.learning == 'labeled' else len(self.unlabeled_mask) # 10 * len(ids) or 75K * len(ids) 
        return num_pixels * len(self.img_ids)
    
    def __getitem__(self, index):
        img_idx, px_idx = index % len(self.img_ids), index // len(self.img_ids)
        if self.fold_setup == 'spatial':
            if self.learning == 'labeled':
                px_idx = self.ids[px_idx]                                       # px_idx returns [0, len) but self.ids are not continuous, so get that px_id from self.ids
        else:
            img_idx = self.img_ids[img_idx]                                     # img_idx are in [0, len(img_ids)) but self.img_ids are not contigous, so get that img_idx from self.img_ids
            img_idx = self.glob_img_id_dict[img_idx]                            # Get global image index.
        print('px_idx: {}, img_idx: {}'.format(px_idx, img_idx))
        return self._get_sample(img_idx=img_idx, px_idx=px_idx)
            
    
if __name__ == "__main__":
    fold_setups = ['temporal_day', 'temporal_year']
    learnings = ['labeled', 'unlabeled']
    for f in fold_setups:
        for l in learnings:
            ids = None if f == 'spatial' and l == 'unlabeled' else [2, 1]
            dataset = Lake2dFoldDataset(learning=l, date_type='month', fold_setup=f, ids=ids)
            print('learning: {}, setup: {}, len: {}'.format(l, f, len(dataset)))
            patch, date_type, reg_val, (img_idx, px, py) = dataset[19]
            
    # dataset = Lake2dFoldDataset(learning='labeled', date_type='month', fold_setup='spatial', ids=[0, 1, 2])
    # print('len:', len(dataset))
    # patch, date_type, reg_val, (img_idx, px, py) = dataset[71]
    