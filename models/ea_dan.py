#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 15:42:09 2021

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
from models import EASeq
# from print_params import count_parameters


class EADAN(nn.Module):
    """
    
    Args:
        in_channels (int): Number of channels of a sample.
        num_classes (int): Number of classes within the dataset. Used in classifier part.
        split_layer (int): Index of the layer that will be used to create shared feature extractor 
        starting from the beginning.
    """
    def __init__(self, in_channels, num_classes, split_layer, patch_size):
        super(EADAN, self).__init__()
        self._verify(split_layer)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.split_layer = split_layer
        self.ps = patch_size
        
        """ Feature extractor """
        self.feature = self._create_model(start=0, end=self.split_layer, 
                                          use_dropout=False, num_convs=4)
        
        """ Regressor """
        self.regressor = self._create_model(start=self.split_layer, end=None, 
                                            use_dropout=False, num_convs=4)     # One more conv to regressor
        
        """ Classifier """
        self.classifier = self._create_model(start=self.split_layer, end=None, 
                                             num_classes=self.num_classes,
                                             use_dropout=False, num_convs=4)
        
        # No need to init weights since they are already init within EASeq.
    
    def forward(self, x):
        x = self.feature(x)
        reg_out = self.regressor(x)
        class_out = self.classifier(x)
        return reg_out, class_out
    
    """
    Creates model
    """
    def _create_model(self, start, end, num_classes=None, use_dropout=False, num_convs=4):
        return nn.Sequential(*list(EASeq(in_channels=self.in_channels, 
                                         num_classes=num_classes,
                                         use_dropout=use_dropout,
                                         num_convs=num_convs,
                                         patch_size=self.ps).children())[start:end])
        
    """
    Checks split layer is valid
    """
    def _verify(self, split_layer):
        
        len_easeq = len([*EASeq(in_channels=12).children()])
        if split_layer > len_easeq - 1 or split_layer < 1:
            raise Exception("EASeq has {} layers, therefore split layer can be in [1, {}]".format(len_easeq, len_easeq - 1))
            
if __name__ == "__main__":
    in_channels, num_classes = 12, 12
    patch_size = 3
    split_layer = 4
    model = EADAN(in_channels=in_channels, num_classes=num_classes, split_layer=split_layer, patch_size=patch_size)
    model_total_params = sum(p.numel() for p in model.parameters())
    model_trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total: {}, trainable: {}'.format(model_total_params, model_trainable_total_params))
    # count_parameters(model)
    # inp = torch.randn(2, in_channels, patch_size, patch_size)
    # outp_reg, outp_class = model(inp)
    # print('shapes, reg: {}, class: {}'.format(outp_reg.shape, outp_class.shape))
