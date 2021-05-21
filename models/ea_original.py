#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:04:00 2021

@author: melike
"""

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import time
import sys
sys.path.append("..")
import constants as C

class EAOriginal(nn.Module):
    def __init__(self, in_channels, patch_size):
        super(EAOriginal, self).__init__()
        self.__verify(in_channels=in_channels)
        self.patch_size = patch_size
        
        """ Convolution layers """
        self.conv1 = self.__make_layer(in_channels=in_channels, out_channels=64)
        self.conv2 = self.__make_layer(in_channels=64, out_channels=64)
        self.conv3 = self.__make_layer(in_channels=64, out_channels=64)
        self.conv4 = self.__make_layer(in_channels=64, out_channels=64)
        
        """ FC layers """
        self.fc1 = nn.Linear(in_features=64 * self.patch_size * self.patch_size * 12,  # A sample is (1, 12, 9) or (1, 20, 15)
                             out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        
        """ Init all layer's weight & bias """
        self.__init_weight_bias()
        
    def __verify(self, in_channels):
        if in_channels != 1:
            raise Exception('Only works with 1 channel patches.')
            
    def __make_layer(self, in_channels, out_channels):
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=3, padding=1)                                   # Padding is 1 due to kernel_size is 3, so 3 // 2 = 1. 
        layers = [conv2d, nn.Tanh()]
        return nn.Sequential(*layers)
    
    def __init_weight_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = torch.flatten(x, 1)                                                        # Flatten all dimensions except batch
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
                

if __name__ == "__main__":
    in_channels, patch_size = 1, 3
    model = EAOriginal(in_channels=in_channels, patch_size=3)
    inp = torch.randn(2, 1, 12, 9)
    outp = model(inp)
    print('outp.shape:', outp.shape)