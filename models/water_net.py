#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:57:36 2021

@author: melike
"""
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import sys
sys.path.append("..")
import constants as C
from datasets import Lake2dDataset

class WaterNet(nn.Module):
    def __init__(self, in_channels, patch_size):
        super(WaterNet, self).__init__()
        self.__verify(in_channels, patch_size)
        self.in_channels = in_channels
        self.ps = patch_size
        
        padding2 = (0, 1, 1) if self.ps == 5 else (0, 0, 0)
        depth2 = (self.in_channels - 2) * 3
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=3, kernel_size=(3, 1, 1))              # (bs, 1, 12, 5, 5) -> (bs, 3, 10, 5, 5)
        self.bn = nn.BatchNorm3d(depth2)
        self.conv2 = nn.Conv3d(in_channels=1, out_channels=10, 
                               kernel_size=(depth2, 3, 3), padding=padding2)                      # (bs, 1, 30, 5, 5) -> (bs, 10, 1, 5, 5)
        self.conv3 = nn.Conv3d(in_channels=1, out_channels=5, kernel_size=(10, 3, 3))             # (bs, 1, 10, 5, 5) -> (bs, 5, 1, 3, 3)
        
        self.fc1 = nn.Linear(in_features=45, out_features=9)                                      # (bs, 45) -> (bs, 9)
        self.fc2 = nn.Linear(in_features=9, out_features=1)                                       # (bs, 9) -> (bs, 1)
        
        
    def __verify(self, in_channels, patch_size):
        if in_channels < 2:
            raise Exception('in_channels should be greater than 2 for WaterNet!')
        if patch_size not in [5, 7]:
            raise Exception('WaterNet convolution kernels only work for patch_size={5, 7}.')
        
    """
    Changes indices 1 and 2 of input to use Batchnorm. 
    """
    def __change_index(self, x):
        return x.view(x.shape[0], x.shape[2], x.shape[1], x.shape[3], x.shape[4])
    
    """
    Takes a 3D input, reshapes it
    """
    def _reshape(self, inp):
        outp_shape = (inp.shape[0], -1, inp.shape[3], inp.shape[4])
        outp = torch.reshape(inp, outp_shape)                                   # (bs, 3, 14, 7, 7) -> (bs, 42, 7, 7)
        return torch.unsqueeze(outp, dim=1)                                     # (bs, 42, 7, 7) -> (bs, 1, 42, 7, 7) to use Conv3d
    
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)                                           # Add dim to make input work with conv3d.
        # print('input: {}'.format(x.shape))
        x = self._reshape(F.relu(self.conv1(x)))
        # print('conv1: {}'.format(x.shape))
        x = self.__change_index(self.bn(self.__change_index(x)))
        # print('bn: {}'.format(x.shape))
        x = self._reshape(F.relu(self.conv2(x)))
        # print('conv2: {}'.format(x.shape))
        x = self._reshape(F.relu(self.conv3(x)))
        # print('conv3: {}'.format(x.shape))
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.fc1(x))
        # print('fc1: {}'.format(x.shape))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    in_channels, patch_size = 12, 5                                             # Depth and channels are different due to convolution on channels.
    model = WaterNet(in_channels=in_channels, patch_size=patch_size)
    # inp = torch.randn(2, in_channels, patch_size, patch_size)
    # outp = model(inp)
    
    date_type = 'year'
    is_orig_model = False
    labeled_2d_dataset = Lake2dDataset(learning='labeled', date_type=date_type, patch_size=patch_size, is_orig_model=is_orig_model)
    labeled_args = {'batch_size': C.BATCH_SIZE,                                              # 12 in SegNet paper
                'shuffle': False,
                'num_workers': 4}
    labeled_loader = DataLoader(labeled_2d_dataset, **labeled_args)
    it = iter(labeled_loader)
    first = next(it)
    patch, date_type, reg_val, (img_idx, px, py) = first
    outp = model(patch)
    print(outp.shape)

