#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 16:14:08 2021

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
from models import VerboseExecution

cfg = [64, 64, 64, 64]

class EASeq(nn.Module):
    def __init__(self, in_channels, num_classes=None, use_dropout=False, num_convs=4, patch_size=3):
        super(EASeq, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.num_convs = num_convs - 1                                                # - 1 due to creating 1st conv layer separately. 
        self.patch_size = patch_size
        
        self.conv1 = self.make_layer(in_channels=self.in_channels, out_channels=64,   # Only 1st conv's kernel size changes to meet different patch sizes.  
                                     kernel_size=self.patch_size)
        # self.conv2 = self.make_layer(in_channels=64, out_channels=64)
        # self.conv3 = self.make_layer(in_channels=64, out_channels=64)
        # self.conv4 = self.make_layer(in_channels=64, out_channels=64)
        self.make_layers()
        
        self.fc1 = nn.Sequential(nn.Flatten(start_dim=1),
                                 nn.Linear(in_features=64 * 3 * 3, out_features=128), # Still 3x3 since all convs except conv1 has 3x3 kernel. 
                                 nn.Tanh(), nn.Dropout2d())
        
        out_features = 1 if self.num_classes is None else self.num_classes
        self.fc2 = nn.Linear(in_features=128, out_features=out_features)
        
        """ Init weights """
        self._init_weights()
        
    def make_layers(self):
        """ Using nn.Sequential """
        self.convs = []
        for i in range(self.num_convs):
            dropout_now = self.use_dropout and i == self.num_convs - 1                # Dropout only on last layer
            conv = self.make_layer(in_channels=64, out_channels=64, 
                                   use_dropout=dropout_now, kernel_size=3)            # Only conv1's kernel_size changes to meet differet patch sizes. 
            self.convs.append(conv)
            setattr(self, 'conv{}'.format(i + 2), conv)                               # They start from 2, so add 2. 
        # """ Using nn.ModuleList """
        # self.convs = nn.ModuleList([self.make_layer(in_channels=64, out_channels=64) for i in range(self.num_convs)])
    
    def make_layer(self, in_channels, out_channels, kernel_size, use_dropout=False):
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1)
        layers = [nn.ReflectionPad2d(padding=1), conv2d, nn.Tanh()]
        if use_dropout:
            layers += [nn.Dropout2d()]
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.conv1(x)
        ## x = self.conv2(x)
        ## x = self.conv3(x)
        ## x = self.conv4(x)
        """ Using nn.Sequential """
        for conv in self.convs:
            x = conv(x)
        # """ Using nn.ModuleList """
        # for conv in self.convs:
        #     x = conv(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):            # glorot_uniform in Keras is xavier_uniform_  
                nn.init.xavier_uniform_(m.weight)
    
if __name__ == "__main__":
    # in_channels, num_classes = 12, 4
    in_channels = 12
    patch_size = 11
    model = EASeq(in_channels=in_channels, num_convs=5, use_dropout=True, patch_size=patch_size)
    inp = torch.randn(2, in_channels, patch_size, patch_size)
    # verbose_model = VerboseExecution(model=model)
    # _ = verbose_model(inp)
    print(model(inp).shape)