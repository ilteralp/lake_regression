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
import itertools
import sys
sys.path.append("..")
import constants as C
from models import EAOriginal, EASeq
# from print_params import count_parameters

class EAOriginalDAN(nn.Module):
    def __init__(self, in_channels, patch_size, split_layer, num_classes, use_atrous_conv=False, reshape_to_mosaic=False):
        super(EAOriginalDAN, self).__init__()
        self.__verify(in_channels=in_channels, split_layer=split_layer, patch_size=patch_size,
                      use_atrous_conv=use_atrous_conv, reshape_to_mosaic=reshape_to_mosaic)
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.split_layer = split_layer
        self.num_classes = num_classes
        self.use_atrous_conv = use_atrous_conv
        self.reshape_to_mosaic = reshape_to_mosaic
        
        """ Feature Extractor """
        self.feature = self.create_model(start=0, end=split_layer)
        
        """ Regressor """
        self.regressor = self.create_model(start=self.split_layer, end=None)
        
        """ Classifier """
        self.classifier = self.create_model(start=self.split_layer, end=None, num_classes=self.num_classes)
     
    """
    Check in_channels and split_layer are valid. 
    """
    def __verify(self, in_channels, patch_size, split_layer, use_atrous_conv, reshape_to_mosaic):
        len_ea_org = len([*EAOriginal(in_channels=in_channels, patch_size=patch_size, 
                                      use_atrous_conv=use_atrous_conv, 
                                      reshape_to_mosaic=reshape_to_mosaic).children()])
        if split_layer > len_ea_org - 1 or split_layer < 1:
            raise Exception('EAOriginal has {} layers, therefore split layer can be in [1, {}]. Given: {}'.format(len_ea_org, len_ea_org - 1, split_layer))
        
    """
    Create multi-task model. 
    """
    def create_model(self, start, end, num_classes=1):
        return nn.Sequential(*list(EAOriginal(in_channels=self.in_channels, 
                                              patch_size=self.patch_size,
                                              num_classes=num_classes,
                                              use_atrous_conv=self.use_atrous_conv,
                                              reshape_to_mosaic=self.reshape_to_mosaic).children())[start:end])
    
    def forward(self, x):
        x = self.feature(x)
        reg_out = self.regressor(x)
        class_out = self.classifier(x)
        return reg_out, class_out
    
if __name__ == "__main__":
    patch_size, num_classes, split_layer, num_samples = 3, 12, 5, 2
    atrous_convs = [False]
    shapes = [False]
    
    for (use_atrous_conv, reshape_to_mosaic) in itertools.product(atrous_convs, shapes):
        if use_atrous_conv and not reshape_to_mosaic: continue
        print('use_atrous_conv: {}, reshape_to_mosaic: {}'.format(use_atrous_conv, reshape_to_mosaic))
        if reshape_to_mosaic: 
            in_channels = 1
            inp = torch.randn(num_samples, in_channels, 12, 9)
        else:
            in_channels = 12
            inp = torch.randn(num_samples, in_channels, patch_size, patch_size)
    
        model = EAOriginalDAN(in_channels=in_channels, patch_size=patch_size, 
                              split_layer=split_layer, num_classes=num_classes,
                              use_atrous_conv=use_atrous_conv,
                              reshape_to_mosaic=reshape_to_mosaic)
        
        outp = model(inp)
        # count_parameters(model)
        print('')
        print(model)
        print('=' * 72)
        
    # print(model)
    # model_trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('split_layer: {}, trainable: {}'.format(split_layer, model_trainable_total_params))
    # print('=' * 72)

    # count_parameters(model)
    
    # feat_ext = model.create_model(start=0, end=split_layer, num_classes=1)
    # reg = model.create_model(start=split_layer, end=None, num_classes=1)
    # clas = model.create_model(start=split_layer, end=None, num_classes=num_classes)
    
    
    
    
    
    