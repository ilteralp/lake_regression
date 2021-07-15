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
import itertools
import sys
sys.path.append("..")
import constants as C
# from print_params import count_parameters


class EAOriginal(nn.Module):
    def __init__(self, in_channels, patch_size, num_classes=1, use_atrous_conv=False, reshape_to_mosaic=False):
        super(EAOriginal, self).__init__()
        self.__verify(in_channels=in_channels, use_atrous_conv=use_atrous_conv, reshape_to_mosaic=reshape_to_mosaic)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.use_atrous_conv = use_atrous_conv
        self.depth = 12 if reshape_to_mosaic else 1
        
        """ Convolution layers & 1st FC """
        if not self.use_atrous_conv:
            self.conv1 = self.__make_layer(in_channels=in_channels, out_channels=64)
            self.conv2 = self.__make_layer(in_channels=64, out_channels=64)
            self.conv3 = self.__make_layer(in_channels=64, out_channels=64)
            self.conv4 = self.__make_layer(in_channels=64, out_channels=64)
            self.fc1 = self.__make_first_fc(in_features=64 * self.patch_size * self.patch_size * self.depth, out_features=128)  # input/output is 12*patch_size^2. 
            
        else:
            self.conv1 = self.__make_atr_conv_layer(in_channels=in_channels, out_channels=64, dilation=3, kernel_size=3)
            self.conv2 = self.__make_atr_conv_layer(in_channels=64, out_channels=64, dilation=2, kernel_size=3)
            self.conv3 = self.__make_atr_conv_layer(in_channels=64, out_channels=64, dilation=(2, 1), kernel_size=3)
            self.conv4 = self.__make_atr_conv_layer(in_channels=64, out_channels=64, dilation=1, kernel_size=3)
            self.fc1 = self.__make_first_fc(in_features=64*4*3, out_features=128)
            
        
        """ FC layers """
        # self.fc1 = nn.Linear(in_features=64 * self.patch_size * self.patch_size * 12,  # A sample is (1, 12, 9) or (1, 20, 15)
        #                      out_features=128)
        # self.fc1 = nn.Sequential(nn.Flatten(start_dim=1),
        #                          nn.Linear(in_features=64 * self.patch_size * self.patch_size * 12, out_features=128), 
        #                          nn.Tanh())
        self.fc2 = nn.Linear(in_features=128, out_features=self.num_classes)
        
        """ Init all layer's weight & bias """
        self.__init_weight_bias()
        
    def __verify(self, in_channels, use_atrous_conv, reshape_to_mosaic):
        if use_atrous_conv and not reshape_to_mosaic:
            raise Exception('Input patches should be mosaic-shaped to use atrous convolutions.')
        
        if reshape_to_mosaic:
            if in_channels != 1:
                raise Exception('Mosaic-shaped patches should have 1 channel. Given: {}'.format(in_channels))
        else:                                                                          # Regular-shaped input should be (2, 12, bs, bs) 
            if in_channels != 12:
                raise Exception('Model works for patches with 12 channels. Given: {}'.format(in_channels))
            
    def __make_layer(self, in_channels, out_channels):
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=1)                                   # Padding is 1 due to kernel_size is 3, so 3 // 2 = 1. 
                           # kernel_size=3, padding=1
        layers = [conv2d, nn.Tanh()]
        return nn.Sequential(*layers)
    
    def __make_first_fc(self, in_features, out_features):
        fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        layers = [nn.Flatten(start_dim=1), fc1, nn.Tanh()]
        return nn.Sequential(*layers)
    
    def __make_atr_conv_layer(self, in_channels, out_channels, dilation, kernel_size):
        conv2d_dil = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               dilation=dilation, kernel_size=kernel_size, padding=1)
        layers = [conv2d_dil, nn.Tanh()]
        return nn.Sequential(*layers)
    
    def __init_weight_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
                
    def forward(self, x):
        # print('inp: {}'.format(x.shape))
        x = self.conv1(x)
        # print('conv1: {}'.format(x.shape))
        x = self.conv2(x)
        # print('conv2: {}'.format(x.shape))
        x = self.conv3(x)
        # print('conv3: {}'.format(x.shape))
        x = self.conv4(x)
        # print('conv4: {}'.format(x.shape))
        
        x = self.fc1(x)
        # print('fc1: {}'.format(x.shape))
        x = self.fc2(x)
        return x
                

if __name__ == "__main__":
    # in_channels, patch_size = 12, 3
    patch_size = 1
    num_samples = 2
    # model_for_mosaic = EAOriginal(in_channels=in_channels, patch_size=patch_size, 
    #                               use_atrous_conv=True, reshape_to_mosaic=True)
    # inp_mosaic = torch.randn(2, in_channels, 12, 9)
    
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
            
            
        model = EAOriginal(in_channels=in_channels, patch_size=patch_size, 
                           use_atrous_conv=use_atrous_conv, 
                           reshape_to_mosaic=reshape_to_mosaic)
        outp = model(inp)
        # print(model)
        # print('')
        # count_parameters(model)
        print('=' * 72)
    
    # model_total_params = sum(p.numel() for p in model.parameters())
    # model_trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('total: {}, trainable: {}'.format(model_total_params, model_trainable_total_params))
    # print(count_parameters(model))