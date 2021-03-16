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
from verbose_execution import VerboseExecution

class EANet(nn.Module):
    def __init__(self, in_channels, num_classes=None):
        super(EANet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        self.pad = nn.ReflectionPad2d(padding=1)
        self.act = nn.Tanh()
        self.conv_first = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=3, stride=1)
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=128)
        out_features = 1 if self.num_classes is None else self.num_classes
        self.fc2 = nn.Linear(in_features=128, out_features=out_features)
        
        """ Init weights """
        self._init_weights()
        
    def forward(self, x):
        x = self.act(self.conv_first(self.pad(x)))                              # (bs, 12, 3, 3) -> (bs, 64, 3, 3)
        x = self.act(self.conv(self.pad(x)))                                    # (bs, 64, 3, 3) -> (bs, 64, 3, 3)
        x = self.act(self.conv(self.pad(x)))                                    # (bs, 64, 3, 3) -> (bs, 64, 3, 3)
        x = self.act(self.conv(self.pad(x)))                                    # (bs, 64, 3, 3) -> (bs, 64, 3, 3)
        
        x = self.act(self.fc1(x.view(x.shape[0], -1)))                          # (bs, 64 * 3 * 3) -> (bs, 128)
        x = self.fc2(x)                                                         # (bs, 128) -> (bs, 1)
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):            # glorot_uniform in Keras is xavier_uniform_  
                nn.init.xavier_uniform_(m.weight)
    
# def _init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):                    
#         nn.init.xavier_uniform_(m.weight)
        
# def hook_fn(m, i, o):
#   # visualisation[m] = o
#   print(m._get_name())

# def get_all_layers(net):
#   for name, layer in net._modules.items():
#     #If it is a sequential, don't register a hook on it
#     # but recursively register hook on all it's module children
#     if isinstance(layer, nn.Sequential):
#         get_all_layers(layer)
#     else:
#         # it's a non sequential. Register a hook
#         layer.register_forward_hook(hook_fn)
    
if __name__ == "__main__":
    in_channels = 12
    net = EANet(in_channels=in_channels)
    verbose_net = VerboseExecution(net)
    # net.apply(weights_init)                                                     # Init weights
    
    inp = torch.randn(2, 12, 3, 3)
    _ = verbose_net(inp)
    # outp = net(inp)
    # print(outp.shape)
    
    # visualisation = {}
    # get_all_layers(net)
    # outp = net(inp)
    ## print(visusalisation.keys())
    
"""
1. Add inits. 
"""
        
        
        
        
        
        
        