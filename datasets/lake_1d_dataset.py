#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 22:27:59 2021

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

class Lake1dDataset(BaseLakeDataset):
    """
    Balik Lake dataset with pixels as samples. Will be used with pseudo-labeling.
    
    Args:
        learning (string): Type of samples. Should be one of {'labeled', 'unlabeled'}. 
    """
    def __init__(self, learning):
        BaseLakeDataset.__init__(self, learning)
        
        if learning.lower() == 'unlabeled':
            self.unlabeled_mask = self._init_mask()
    
    """
    Returns unlabeled sample mask in 3D to mask 3D data. 
    """
    def _init_mask(self):
        # return torch.from_numpy(self.unlabeled_mask).unsqueeze(0).expand([12, 650, 650])   # Convert to 12x650x650 for comparison.
        return torch.nonzero(torch.from_numpy(self.unlabeled_mask))
            
    """
    A sample is 1x12. 
    """
    def __getitem__(self, index):
        img_idx, px_idx = index % 32, index // 32
        img_path = osp.join(C.IMG_DIR_PATH, self.img_names[img_idx], 'level2a.h5')
        if osp.exists(img_path):
            with h5py.File(img_path, 'r') as img:
                data = img['bands'][:]                                          # (12, 650, 650), ndarray
                if data.shape != (12, 650, 650):
                    raise Exception('Expected input shape to be (12, 650, 650) for {}'.format(img_path))
                data = torch.from_numpy(data.astype(np.int32))                  # Pytorch cannot convert uint16.
                reg_val = None
                if self.learning == 'unlabeled':
                    # masked = data[self.unlabeled_mask]                          # Apply mask
                    # px_val = masked.view(-1, 12)[px_idx]                        # Reshape and retrieve index. 
                    px, py = self.unlabeled_mask[px_idx].numpy()
                else:
                    px, py = C.LABELED_INDICES[0][px_idx], C.LABELED_INDICES[1][px_idx]
                    reg_val = self._get_regression_val(img_idx, px_idx)
                    
                px_val = data[:, px, py].unsqueeze(0)                       # Reshape to 1x12.
                month, season, year = self.dates[img_idx].values()
                # print('\npixel: ({}, {}) at image {}:\n\t{}'.format(px, py, self.img_names[img_idx], px_val))
                return px_val, month, season, year, reg_val
        else:
            raise Exception('Image not found on {}'.format(img_path))
                
if __name__ == "__main__":
    sup_px_idx = -1
    sup_1d_dataset = Lake1dDataset(learning='labeled')
    unsup_1d_dataset = Lake1dDataset(learning='unlabeled')
    px_val, month, season, year, reg_val = sup_1d_dataset[sup_px_idx]
    
    # """ =============== Comparing with h5py indexing =============== """
    # h_img_name = '34'
    # filepath = osp.join(C.ROOT_DIR, 'balik', h_img_name, 'level2a.h5')
    # f = h5py.File(filepath, 'r')
    # img = f['bands']
    # p1, p2 = C.LABELED_INDICES[0][-1], C.LABELED_INDICES[1][-1]
    # print('\nvs\n\npixel: ({}, {}) at image {}:'.format(p1, p2, h_img_name))
    # arr_val = img[:, p1, p2]
    # print('\t', arr_val)