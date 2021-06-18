#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 22:59:33 2021

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
from models import EAOriginal, EASeq

class EAOriginalDAN(nn.Module):
    def __init__(self, in_channels, patch_size, split_layer, num_classes):
        super(EAOriginalDAN, self).__init__()
        self.__verify(in_channels=in_channels, split_layer=split_layer, patch_size=patch_size)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.split_layer = split_layer
        self.num_classes = num_classes
        
        """ Feature Extractor """
        self.feature = self.create_model(start=0, end=split_layer)
        
        """ Regressor """
        self.regressor = self.create_model(start=self.split_layer, end=None)
        
        """ Classifier """
        self.classifier = self.create_model(start=self.split_layer, end=None, num_classes=self.num_classes)
     
    """
    Check in_channels and split_layer are valid. 
    """
    def __verify(self, in_channels, patch_size, split_layer):
        if in_channels != 1:
            raise Exception('Only works with 1 channel patches.')
        len_ea_org = len([*EAOriginal(in_channels=in_channels, patch_size=patch_size).children()])
        if split_layer > len_ea_org - 1 or split_layer < 1:
            raise Exception('EAOriginal has {} layers, therefore split layer can be in [1, {}]. Given: {}'.format(len_ea_org, len_ea_org - 1, split_layer))
        
    """
    Create multi-task model. 
    """
    def create_model(self, start, end, num_classes=1):
        return nn.Sequential(*list(EAOriginal(in_channels=self.in_channels, 
                                              patch_size=self.patch_size,
                                              num_classes=num_classes).children())[start:end])
    
    def forward(self, x):
        x = self.feature(x)
        reg_out = self.regressor(x)
        class_out = self.classifier(x)
        return reg_out, class_out
    
if __name__ == "__main__":
    in_channels, patch_size, num_classes = 1, 3, 12
    for split_layer in range(1, 7):
        model = EAOriginalDAN(in_channels=in_channels, patch_size=patch_size, 
                              split_layer=split_layer, num_classes=num_classes)
        print(model)
        print('=' * 72)
    
    # feat_ext = model.create_model(start=0, end=split_layer, num_classes=1)
    # reg = model.create_model(start=split_layer, end=None, num_classes=1)
    # clas = model.create_model(start=split_layer, end=None, num_classes=num_classes)
    
    
    
    
    
    