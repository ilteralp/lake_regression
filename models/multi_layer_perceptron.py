#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:39:22 2021

@author: melike
"""

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
import constants as C
from print_params import count_parameters

class MultiLayerPerceptron(nn.Module):
    def __init__(self, in_features, cfg):
        super(MultiLayerPerceptron, self).__init__()
        self.in_features = in_features
        self.cfg = cfg
        
        in_features = self.__make_layers()
        self.out = nn.Linear(in_features=in_features, out_features=1)
        
    def __make_fc(self, in_features, out_features):
        fc = nn.Linear(in_features=in_features, out_features=out_features)
        layers = [fc, nn.Tanh()]
        return nn.Sequential(*layers)
    
    def __make_layers(self):
        self.fcs = []
        in_features = self.in_features
        for i, out_features in enumerate(self.cfg):
            fc = self.__make_fc(in_features=in_features, out_features=out_features)
            self.fcs.append(fc)
            setattr(self, 'fc{}'.format(i+1), fc)
            in_features = out_features
        return out_features
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)                                       # (bs, 12, 5, 5) -> (bs, 300)
        for fc in self.fcs:
            x = fc(x)
        
        x = self.out(x)
        return x

if __name__ == "__main__":
    in_channels, patch_size = 12, 5
    in_features = in_channels * patch_size * patch_size
    inp = torch.randn(2, in_channels, patch_size, patch_size)
    for i in range(1, 5):
        model = MultiLayerPerceptron(in_features=in_features, cfg=C.MLP_CFGS['{}_hidden_layer'.format(i)])
        outp = model(inp)
        print(outp.shape)
        count_parameters(model)
        print('*' * 72)