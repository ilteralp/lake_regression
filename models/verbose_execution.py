#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 16:38:27 2021

@author: melike
"""

import torch
import torch.nn as nn
from torchvision import models

class VerboseExecution(nn.Module):
    """
    Prints model execution based on 
    https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904
    
    Args:
        model: A PyTorch nn.module
    """
    def __init__(self, model):
        super(VerboseExecution, self).__init__()
        self.model = model
        
        for layer in self.model.modules():                                      # Register a hook for each layer
            layer.register_forward_hook(
                lambda layer, _, outp: print('{}'.format(layer._get_name()))
                # lambda layer, _, outp: print('{}: {}'.format(layer._get_name(), outp.shape))
            )
        
    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    resnet = models.resnet50(pretrained=False)
    verbose_resnet = VerboseExecution(resnet)
    inp = torch.randn(2, 3, 224, 244)
    _ = verbose_resnet(inp)