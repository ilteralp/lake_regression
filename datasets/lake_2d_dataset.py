#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:12:18 2021

@author: melike
"""

import torch
import numpy as np
import h5py
import os.path as osp
import sys
sys.path.append("..")
import constants as C
from base_lake_dataset import BaseLakeDataset

class Lake2dDataset(BaseLakeDataset):
    """
    Balik Lake dataset with patches as samples. Will be used with DAN. 
    
    Args:
        learning (string): Type of samples. Should be one of {'labeled', 'unlabeled'}. 
        patch_size (int): Size of the patch that is centered on the sample pixel. Should be an odd number. 
    """
    
    def __init__(self, learning, patch_size=3):
        BaseLakeDataset.__init__(self, learning)
        
        self.patch_size = patch_size
        if learning.lower() == 'unlabeled':
            self.unlabeled_mask = self._init_mask()
            
    """
    Returns unlabeled sample mask that keeps unlabeled sample indices. 
    """
    def _init_mask(self):
        return torch.nonzero(torch.from_numpy(self.unlabeled_mask))
    
    """
    A sample is 12x3x3. 
    """
    def __getitem__(self, index):
        img_idx, px_idx = index % 32, index // 32
        img_path = osp.join(C.IMG_DIR_PATH, self.img_names[img_idx], 'level2a.h5')
        if osp.exists(img_path):
            with h5py.File(img_path, 'r') as img:
                data = img['bands'][:]                                          # (12, 650, 650), ndarray
                if data.shape != (12, 650, 650):
                        raise Exception('Expected input shape to be (12, 650, 650) for {}'.format(img_path))
                # No need for padding since all lake pixels are not in borders :) 
                # if self.patch_size is not None:                                # Apply padding to crop patches from border pixels.
                #     pad = self.patch_size // 2
                #     data = np.pad(data, ((0, 0), (pad, pad), (pad, pad)), mode='symmetric')
                #     print('Padded, data.shape:', data.shape)
                #     if self.learning == 'unlabeled':
                #         self.unlabeled_mask = np.pad(self.unlabeled_mask, ((0, 0), (pad, pad), (pad, pad)), mode='symmetric')
                #         print('Padded, unlabeled_mask.shape:', self.unlabeled_mask.shape)
                # If you decide back to padding, don't forget to add pad value to each pixel. 
                data = torch.from_numpy(data.astype(np.int32))                  # Pytorch cannot convert uint16
                reg_val = None
                if self.patch_size is not None:
                    pad = self.patch_size // 2
                if self.learning == 'unlabeled':
                    px, py = self.unlabeled_mask[px_idx].numpy()                # Check for cuda
                else:                                                           # Labeled
                    px, py = C.LABELED_INDICES[0][px_idx], C.LABELED_INDICES[1][px_idx]
                    reg_val = self._get_regression_val(img_idx, px_idx)
                    
                patch = data[:, px-pad : px+pad + 1, py-pad : py+pad+1]         # (12, 3, 3)
                month, season, year = self.dates[img_idx].values()
                return patch, month, season, year, reg_val
                    
        else:
            raise Exception('Image not found on {}'.format(img_path))
     
if __name__ == "__main__":
    # filepath = osp.join(C.ROOT_DIR, 'balik', '2', 'level2a.h5')
    # f = h5py.File(filepath, 'r')
    # print("Keys: %s" % f.keys())
    # print('shape:', f['bands'].shape)
    # img = f['bands']
    # for i in range(10):
    #     px, py = C.LABELED_INDICES[0][i], C.LABELED_INDICES[1][i]
    #     print('({}, {}): {}'.format(px, py, img[:, px, py]))
    #     print('({}, {}): {}'.format(py, px, img[:, py, px]))
    #     print('=' * 20)
    # # print('\n Regression value:', 86.14)
    
    # img = io.imread(C.MASK_PATH)
    # lake_mask = np.all(img == (255, 0, 255), axis=-1) # Includes labeled pixels too! It must be 
    # print('Total lake pixels: {}'.format(np.sum(lake_mask)))
    
    sup_2d_lake_dataset = Lake2dDataset(learning='labeled', patch_size=3)
    # unsup_lake_dataset = Lake2dDataset(learning='unlabeled')
    unsup_2d_lake_dataset = Lake2dDataset(learning='unlabeled', patch_size=3)
    patch, month, season, year, reg_val = sup_2d_lake_dataset[0]