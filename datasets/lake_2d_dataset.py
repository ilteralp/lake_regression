#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:12:18 2021

@author: melike
"""

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
import os
import os.path as osp
import sys
import copy
import csv
from skimage import io
sys.path.append("..")
import constants as C

class Lake2dDataset(Dataset):
    """
    Balik Lake dataset with patches as samples.
    
    Args:
        labeling (string): Should be one of {'labeled', 'unlabeled'}. 
        # sample_type (string): Should be one of {'1D', '2D', '3D'}. For '3D' type, patch_size is expected. 
        patch_size (optional, int): Size of the patch that is centered on the sample pixel.
        # channel_first (optional, boolean): If true, data is read 12x32. Otherwise it's 32x12. 
    """
    
    def __init__(self, learning, patch_size=3):
        self.learning = learning.lower()
        self.patch_size = patch_size
        self.img_names = self._get_image_names()
        self.dates = self._read_date_info()
        # if self.learning == 'unlabeled':
            # self.mask = self._load_mask()
        self.unlabeled_mask = self._load_unlabeled_mask()
        if self.learning == 'labeled':
            self.reg_vals = self._read_GT()
        
    def __len__(self):
        return self.num_samples if self.learning == 'unlabeled' else len(C.LABELED_INDICES[0])
        # Check len of splits in case of labeled samples !
    
    # def __getitem_base__(self, index):
    #     chs = []
    #     for f in os.listdir(C.IMG_DIR_PATH):
    #         img_path = osp.join(C.IMG_DIR_PATH, f, 'level2a.h5')
    #         if osp.exists(img_path):
    #             with h5py.File(img_path, 'r') as img:
    #                 data = img['bands'][:]  # (12, 650, 650)
    #                 if data.shape != (12, 650, 650):
    #                     raise Exception('Expected input shape to be (12, 650, 650) for {}'.format(img_path))
    #                 data = torch.from_numpy(data.astype(np.int32))              # Pytorch cannot convert uint16
    #                 if self.learning == 'unlabeled':
    #                     masked = data[self.unlabeled_mask]                      # Apply mask
    #                     px_val = masked.view(-1, 12)[index]                     # Reshape and retrieve index 
    #                 else:
    #                     px_val = np.transpose(data, (1, 2, 0))[C.LABELED_INDICES][index]
    #                 chs.append(px_val)
    #     return torch.stack(chs, dim=0)                                          # Should be 32x12. 
    
    def _get_image_names(self):
        ls = os.listdir(C.IMG_DIR_PATH)
        num_names = sorted([int(l) for l in ls if l not in['22', '23']])        # Skip empty folders
        return [str(l) for l in num_names]
    
    """
    Use with DAN, has 12x3x3 features. 
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
     
    """
    Reads ground-truth file and keeps regression value of labeled samples. 
    """
    def _read_GT(self):
        if self.learning != 'labeled':
            raise Exception('Only labeled samples have regression values!')
        with open(C.GT_PATH) as f:
            reader = csv.reader(f, delimiter='\t')
            return [float(row[2]) for row in reader]
    
    """
    Returns regression value of for given indices. Only for labeled samples. 
    """
    def _get_regression_val(self, img_idx, px_idx):
        ln = img_idx * len(C.LABELED_INDICES[0]) + px_idx
        if(len(self.reg_vals) < ln):
            raise Exception('Accessing index#{} where total length is {} at {}'.format(ln, len(self.reg_vals), C.GT_PATH))
        return self.reg_vals[ln]
    
    """
    Reads month, season and year info of all images. 
    """
    def _read_date_info(self):
        with open(C.DATE_LABELS_PATH, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            return [{'month' : int(l[1]) - 1, 'season' : C.SEASONS[l[2]],       # Months start from 1.
                     'year' : int(l[3])} for l in reader]

    # def _load_mask(self):
    #     img = io.imread(C.MASK_PATH)
    #     return np.all(img == (255, 0, 255), axis=-1)
    
    def _load_unlabeled_mask(self):
        mask = np.all(io.imread(C.MASK_PATH) == (255, 0, 255), axis=-1)         # Lake mask, 650x650
        self.raw_mask = copy.deepcopy(mask)
        print('before', mask[C.LABELED_INDICES])
        print('before, sum of mask pixels:', np.sum(mask))
        mask[C.LABELED_INDICES] = False                                         # Set labeled indices to false.  
        # print('indices', len(torch.nonzero(torch.from_numpy(mask))))
        self.num_samples = np.sum(mask)
        print('after sum', self.num_samples)
        if self.patch_size is not None:                                         # Patches will be cropped. 
            mask = torch.nonzero(torch.from_numpy(mask))
        else:
            mask = torch.from_numpy(mask).unsqueeze(0).expand([12, 650, 650])   # Convert to 12x650x650 for comparison.
        return mask
    

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