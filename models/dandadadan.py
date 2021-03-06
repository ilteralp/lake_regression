#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 20:35:48 2021

@author: melike

Adapted from https://github.com/fungtion/DANN/blob/master/models/model.py
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

class DandadaDAN(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(DandadaDAN, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        """ Feature extractor """
        self.feature = nn.Sequential()                                          
        self.feature.add_module('f_pad', nn.ReflectionPad2d(padding=2))                         # (bs, 12, 3, 3) -> (bs, 12, 7, 7)
        self.feature.add_module('f_conv1', nn.Conv2d(in_channels=self.in_channels,              # Cannot use padding of Conv2d because it applies zero padding.
                                                     out_channels=64, kernel_size=3, stride=1)) # (bs, 12, 7, 7) -> (bs, 64, 5, 5)
        self.feature.add_module('f_bn1', nn.BatchNorm2d(num_features=64))
        self.feature.add_module('f_tanh1', nn.Tanh())
        self.feature.add_module('f_conv2', nn.Conv2d(in_channels=64, out_channels=50,
                                                     kernel_size=3, stride=1))                  # (bs, 64, 5, 5) -> (bs, 50, 3, 3)
        self.feature.add_module('f_bn2', nn.BatchNorm2d(num_features=50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_tanh2', nn.Tanh())
        
        """ Regression """
        self.regressor = nn.Sequential()
        self.regressor.add_module('r_fc1', nn.Linear(in_features=50 * 3 * 3, out_features=100)) # (bs, 300) -> (bs, 100)
        self.regressor.add_module('r_bn1', nn.BatchNorm1d(100))
        self.regressor.add_module('r_tanh1', nn.Tanh())
        self.regressor.add_module('r_drop1', nn.Dropout())                                      # DAN used Dropout2d but it drops whole channel.
        self.regressor.add_module('r_fc2', nn.Linear(in_features=100, out_features=100))        # (bs, 100) -> (bs, 100)    
        self.regressor.add_module('r_bn2', nn.BatchNorm1d(100))
        self.regressor.add_module('r_tanh2', nn.ReLU())
        self.regressor.add_module('r_fc3', nn.Linear(in_features=100, out_features=1))          # (bs, 100) -> (bs, 1)
        
        """ Classification """
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1', nn.Linear(in_features=50 * 3 * 3, out_features=100)) # (bs, 300) -> (bs, 100)
        self.classifier.add_module('c_bn1', nn.BatchNorm1d(num_features=100))
        self.classifier.add_module('c_tanh1', nn.Tanh())
        self.classifier.add_module('c_fc2', nn.Linear(in_features=100, 
                                                      out_features=self.num_classes))           # (bs, 100) -> (bs, num_class)
    
    def forward(self, x):
        feature = self.feature(x)                                                               # -> (bs, 50, 3, 3)
        feature = feature.view(-1, 50 * 3 * 3)                                                  # Flatten features
        reg_out = self.regressor(feature)
        class_out = self.classifier(feature)
        return reg_out, class_out
        
    
if __name__ == "__main__":
    dan = DandadaDAN(in_channels=12, num_classes=4)
    inp = torch.rand(2, 12, 3, 3)
    reg_out, class_out = dan(inp)
    print('shapes, reg: {} class: {}'.format(reg_out.shape, class_out.shape))

"""
Ideas
1. Which way to init weights ?
2. EA's model as regressor
"""