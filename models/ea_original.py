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
import sys
sys.path.append("..")
import constants as C
# from print_params import count_parameters


class EAOriginal(nn.Module):
    def __init__(self, in_channels, patch_size, num_classes=1, use_atrous_conv=False):
        super(EAOriginal, self).__init__()
        self.__verify(in_channels=in_channels)
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.use_atrous_conv = use_atrous_conv
        
        """ Convolution layers & 1st FC """
        if not self.use_atrous_conv:
            self.conv1 = self.__make_layer(in_channels=in_channels, out_channels=64)
            self.conv2 = self.__make_layer(in_channels=64, out_channels=64)
            self.conv3 = self.__make_layer(in_channels=64, out_channels=64)
            self.conv4 = self.__make_layer(in_channels=64, out_channels=64)
            self.fc1 = self.__make_first_fc(in_features=64 * self.patch_size * self.patch_size * 12, out_features=128)
            
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
        
    def __verify(self, in_channels):
        if in_channels != 1:
            raise Exception('Only works with 1 channel patches.')
            
    def __make_layer(self, in_channels, out_channels):
        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                           kernel_size=3, padding=1)                                   # Padding is 1 due to kernel_size is 3, so 3 // 2 = 1. 
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
        
        # x = torch.flatten(x, 1)                                                        # Flatten all dimensions except batch
        # x = torch.tanh(self.fc1(x))
        x = self.fc1(x)
        # print('fc1: {}'.format(x.shape))
        x = self.fc2(x)
        return x
                

if __name__ == "__main__":
    in_channels, patch_size = 1, 3
    model = EAOriginal(in_channels=in_channels, patch_size=3, use_atrous_conv=True)
    # model_total_params = sum(p.numel() for p in model.parameters())
    # model_trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('total: {}, trainable: {}'.format(model_total_params, model_trainable_total_params))

    # print(count_parameters(model))
    inp = torch.randn(2, 1, 12, 9)
    outp = model(inp)
    print('outp.shape:', outp.shape)