#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:12:18 2021

@author: melike
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import os.path as osp
import sys
from skimage import io
sys.path.append("..")
import constants as C

class LakeMultiImageDataset(Dataset):
    """
    Balik Lake dataset with 10-meter pixel resolution and 32 images. 
    
    Args:
        labeling (string): Should be one of {'labeled', 'unlabeled'}. 
        # sample_type (string): Should be one of {'1D', '2D', '3D'}. For '3D' type, patch_size is expected. 
        # patch_size (optional, int): Size of the patch in case of 3D input. 
        # channel_first (optional, boolean): If true, data is read 12x32. Otherwise it's 32x12. 
    """
    
    def __init__(self, learning):
        self.learning = learning.lower()
        if self.learning == 'unlabeled':
            # self.mask = self._load_mask()
            self.unlabeled_mask = self._load_unlabeled_mask()
        
    def __len__(self):
        return self.num_samples if self.learning == 'unlabeled' else len(C.LABELED_INDICES[0])
        # Check len of splits in case of labeled samples !
    
    def __getitem__(self, index):
        chs = []
        for f in os.listdir(C.IMG_DIR_PATH):
            img_path = osp.join(C.IMG_DIR_PATH, f, 'level2a.h5')
            if osp.exists(img_path):
                with h5py.File(img_path, 'r') as img:
                    data = img['bands'][:]  # (12, 650, 650)
                    if data.shape != (12, 650, 650):
                        raise Exception('Expected input shape to be (12, 650, 650) for {}'.format(img_path))
                    data = torch.from_numpy(data.astype(np.int32))              # Pytorch cannot convert uint16
                    if self.learning == 'unlabeled':
                        masked = data[self.unlabeled_mask]                      # Apply mask
                        px_val = masked.view(-1, 12)[index]                     # Reshape and retrieve index 
                    else:
                        px_val = np.transpose(data, (1, 2, 0))[C.LABELED_INDICES][index]
                    chs.append(px_val)
        return torch.stack(chs, dim=0)                                          # Should be 32x12. 
            
    
    # def _load_mask(self):
    #     img = io.imread(C.MASK_PATH)
    #     return np.all(img == (255, 0, 255), axis=-1)
    
    def _load_unlabeled_mask(self):
        mask = np.all(io.imread(C.MASK_PATH) == (255, 0, 255), axis=-1)         # Lake mask, 650x650
        print('before', mask[C.LABELED_INDICES])
        print('before sum', np.sum(mask))
        self.num_samples = np.sum(mask)
        mask[C.LABELED_INDICES] = False                                         # Set labeled indices to false.  
        mask = torch.from_numpy(mask).unsqueeze(0).expand([12, 650, 650])       # Convert to 12x650x650 for comparison.
        return mask
        

if __name__ == "__main__":
    filepath = osp.join(C.ROOT_DIR, 'balik', '1', 'level2a.h5')
    f = h5py.File(filepath, 'r')
    print("Keys: %s" % f.keys())
    print('shape:', f['bands'].shape)
    
    
    # img = io.imread(C.MASK_PATH)
    # lake_mask = np.all(img == (255, 0, 255), axis=-1) # Includes labeled pixels too! It must be 
    # print('Total lake pixels: {}'.format(np.sum(lake_mask)))
    
    sup_lake_dataset = LakeMultiImageDataset(learning='labeled')
    unsup_lake_dataset = LakeMultiImageDataset(learning='unlabeled')
    a = sup_lake_dataset[0]