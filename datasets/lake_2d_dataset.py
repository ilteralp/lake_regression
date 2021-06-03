#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:12:18 2021

@author: melike
"""

import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import h5py
import os.path as osp
import sys
sys.path.append("..")
import constants as C
from datasets import BaseLakeDataset

class Lake2dDataset(BaseLakeDataset):
    """
    Balik Lake dataset with patches as samples. Will be used with DAN. 
    
    Args:
        learning (string): Type of samples. Should be one of {'labeled', 'unlabeled'}. 
        patch_size (int): Size of the patch that is centered on the sample pixel. Should be an odd number. 
        is_orig_model (bool): Type of model. If true, reshapes patches to 1x12x9, otherwise 12x3x3
        date_type (string): Type of date label that will be used for classification. 
        Should be one of {'month', 'season', 'year'}.
    """
    
    def __init__(self, learning, date_type, is_orig_model, patch_size=3):
        BaseLakeDataset.__init__(self, learning, date_type, patch_size)
        
        self.patch_size = patch_size
        self.is_orig_model = is_orig_model
        if learning.lower() == 'unlabeled':
            self.unlabeled_mask = self._init_mask()
        
        """
        Load all images
        """
        Lake2dDataset.images = []
        for img_name in self.img_names:
            img_path = osp.join(C.IMG_DIR_PATH, img_name, 'level2a.h5')
            if osp.exists(img_path):
                with h5py.File(img_path, 'r') as img:
                    data = img['bands'][:]
                    if data.shape != (12, 650, 650):
                        raise Exception('Expected input shape to be (12, 650, 650) for {}'.format(img_path))
                    data = torch.from_numpy(data.astype(np.float32))                # Pytorch cannot convert uint16
                    Lake2dDataset.images.append(data)
            else:
                raise Exception('Image not found on {}'.format(img_path))

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
        return self._get_sample(img_idx=img_idx, px_idx=px_idx)
    
    """
    Reshapes patch from 12x3x3 to 1x12x9 without messing between channels and
    returns it. 
    """
    def __reshape_patch(self, patch):
        # x = torch.unbind(patch)
        # vs = [torch.cat((x[3*i], x[3*i+1], x[3*i+2]), 1) for i in range(4)]
        # return torch.cat(vs, 0).unsqueeze(0)
        ps = self.patch_size
        return patch.view(4, 3, ps, ps).permute(0, 2, 1, 3).reshape(1, 4 * ps, 3 * ps)
    
    def _get_sample(self, img_idx, px_idx):
        data = Lake2dDataset.images[img_idx]
        pad = self.patch_size // 2
        reg_val = 1.0
        if self.learning == 'unlabeled':
            px, py = self.unlabeled_mask[px_idx].numpy()                # Check for cuda
        else:                                                           # Labeled
            px, py = C.LABELED_INDICES[0][px_idx], C.LABELED_INDICES[1][px_idx]
            reg_val = self._get_regression_val(img_idx, px_idx)
            
        patch = data[:, px-pad : px+pad + 1, py-pad : py+pad+1]         # (12, 3, 3) or (12, 5, 5)
        if self.is_orig_model:                                          # Only reshape for EAOriginal model.     
            patch = self.__reshape_patch(patch=patch)                   # (1, 12, 9) or (1, 20, 15)
        
        date_class = self.dates[img_idx][self.date_type]
        
        if self.patch_means is not None and self.patch_stds is not None: # Normalize patch
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Normalize(
                    mean=self.patch_means,
                    std=self.patch_stds
                    ),
                ])
            patch = transform(patch)
            
        # if self.reg_mean is not None and self.reg_std is not None:    # Standardize regression value. 
        #     reg_val = (reg_val - self.reg_mean) / self.reg_std
        
        if self.reg_min is not None and self.reg_max is not None:       # Normalize regression value. 
            reg_val = self.normalize_reg(reg_val)                    
        
        return patch, date_class, np.expand_dims(reg_val, axis=0).astype(np.float32), (img_idx, px, py)

    """
    Normalizes given regression value to [-1, 1] range. 
    """
    def normalize_reg(self, reg_val):
        reg_val = 2 * (reg_val - self.reg_min) / (self.reg_max - self.reg_min) - 1
        if reg_val > 1.0:    reg_val = 1.0
        elif reg_val < -1.0: reg_val = -1.0
        return reg_val
    
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
    
    # ps, date_type = [5], 'year'
    patch_size, date_type = 3, 'year'
    is_orig_model = False
    # for patch_size in ps:
    labeled_2d_dataset = Lake2dDataset(learning='labeled', date_type=date_type, patch_size=patch_size, is_orig_model=is_orig_model)
    # train_set = Subset(labeled_2d_dataset, indices=[*range(0, 10)])
    # test_set = Subset(labeled_2d_dataset, indices=[*range(10, 20)])
    # val_set = Subset(labeled_2d_dataset, indices=[*range(20, 30)])
    # tr_indices = np.asarray([*range(10, 20)])
    # reg_min, reg_max = get_reg_min_max(labeled_2d_dataset.reg_vals[tr_indices])
    # labeled_2d_dataset.set_reg_min_max(reg_min=reg_min, reg_max=reg_max)
    
    # unlabeled_2d_dataset = Lake2dDataset(learning='unlabeled', date_type=date_type, patch_size=patch_size)
    # print('patch_size: {} lens, l: {}, u: {}'.format(patch_size, len(labeled_2d_dataset), len(unlabeled_2d_dataset)))
    # unlabeled_2d_dataset = Lake2dDataset(learning='unlabeled', date_type='year', patch_size=3)
    patch, date_type, reg_val, (img_idx, px, py) = labeled_2d_dataset[0]
    
    labeled_args = {'batch_size': C.BATCH_SIZE,                                              # 12 in SegNet paper
                    'shuffle': False,
                    'num_workers': 4}
    # patches_mean, patches_std, _, _ = calc_mean_std(DataLoader(labeled_2d_dataset, **labeled_args))

    # # patch, date_type, reg_val, (img_idx, px, py) = labeled_2d_dataset[0]
    # patch, date_type, reg_val, (img_idx, px, py) = labeled_2d_dataset[0]
    # print('reg_val:', reg_val)
    
    # labeled_loader = DataLoader(labeled_2d_dataset, **labeled_args)
    # it = iter(labeled_loader)
    # first = next(it)
    # patch, date_type, reg_val, (img_idx, px, py) = first 
    # # for d in [patch, date_type, reg_val]:
    #     # print('type: {}, dtype: {}, shape: {}'.format(type(d), d.dtype, d.shape))
    
        
    # sec = next(it)
    # print('1st, img, pixels: {}'.format(first[-1]))
    # print('2nd, img, pixels: {}'.format(sec[-1]))
    
    # for e in range(0, 2):
    # for batch_idx, (patch, month, season, year, reg_val, (img_idx, px, py)) in enumerate(labeled_loader):
    #     print('shape, patch: {}, reg_val: {}'.format(patch.shape, reg_val.shape))
            
            # print('batch:', batch_idx)
            # print('pixels: ({}, {})'.format(px, py))
    