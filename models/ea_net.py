#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 22:07:22 2021

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

class EANet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(EANet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.pad = nn.ReflectionPad2d(padding=1)
        self.act = nn.Tanh()
        self.conv_first = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=1)
        
        
    def forward(self, x):
        x = self.act(self.conv_first(self.pad(x)))                              # (bs, 12, 3, 3) -> (bs, 64, 3, 3)
        x = self.act(self.conv(self.pad(x)))                                    # (bs, 64, 3, 3) -> (bs, 64, 3, 3)
        x = self.act(self.conv(self.pad(x)))                                    # (bs, 64, 3, 3) -> (bs, 64, 3, 3)
        x = self.act(self.conv(self.pad(x)))                                    # (bs, 64, 3, 3) -> (bs, 64, 3, 3)
        
        x = self.act(self.fc1(x.view(x.shape[0], -1)))                          # (bs, 64 * 3 * 3) -> (bs, 128)
        x = self.fc2(x)                                                         # (bs, 128) -> (bs, 1)
        return x
    
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):                    # glorot_uniform in Keras is xavier_uniform_  
        nn.init.xavier_uniform_(m.weight)
    
if __name__ == "__main__":
    in_channels, num_classes = 12, 4
    net = EANet(in_channels=in_channels, num_classes=num_classes)
    net.apply(weights_init)                                                     # Init weights
    
    inp = torch.randn(2, 12, 3, 3)
    outp = net(inp)
    print(outp.shape)
    
"""
1. Add inits. 
"""
        
        
        
        
        
        
        