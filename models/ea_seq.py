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
    def __init__(self, in_channels, num_classes=None, use_dropout=False, num_convs=4):
        super(EASeq, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.use_dropout = use_dropout
        self.num_convs = num_convs - 1                                          # - 1 due to creating 1st conv layer separately. 
        
        self.conv1 = nn.Sequential(*self.make_layer(in_channels=self.in_channels, out_channels=64))
        # self.conv2 = self.make_layer(in_channels=64, out_channels=64)
        # self.conv3 = self.make_layer(in_channels=64, out_channels=64)
        # self.conv4 = self.make_layer(in_channels=64, out_channels=64)
        self.convs = self.make_layers()
        
        self.fc1 = nn.Sequential(nn.Flatten(start_dim=1),
                                 nn.Linear(in_features=64 * 3 * 3, out_features=128),
                                 nn.Tanh(), nn.Dropout2d())
        
        out_features = 1 if self.num_classes is None else self.num_classes
        self.fc2 = nn.Linear(in_features=128, out_features=out_features)
        
        """ Init weights """
        self._init_weights()
        
    def make_layers(self):
        # layers = []
        # in_channels = self.in_channels
        # for out_channels in cfg:
        #     conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        #     layers += [nn.ReflectionPad2d(padding=1), conv2d, nn.Tanh()]
        #     in_channels = out_channels
        # return nn.Sequential(*layers)
        layers = []
        for i in range(self.num_convs):
            layers += self.make_layer(in_channels=64, out_channels=64)
        return nn.Sequential(*layers)
    
    def make_layer(self, in_channels, out_channels):
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1)
        layers = [nn.ReflectionPad2d(padding=1), conv2d, nn.Tanh()]
        if self.use_dropout:
            layers += [nn.Dropout2d()]
        # return nn.Sequential(*[nn.ReflectionPad2d(padding=1), conv2d, nn.Tanh()])
        # return nn.Sequential(*layers)
        return layers
        
    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.convs(x)
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
    model = EASeq(in_channels=in_channels, num_convs=5)
    inp = torch.randn(2, in_channels, 3, 3)
    verbose_model = VerboseExecution(model=model)
    _ = verbose_model(inp)
    # print(model(inp).shape)