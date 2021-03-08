
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:40:07 2021

@author: melike
"""

import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import constants as C

class Metrics:
    """
    Measures regression and classification metrics.
    
    Args:
        num_folds (int): Number of folds. Can be None for training without 
        cross-validation or an int >0 for training with cross-validation. 
    """
    
    def __init__(self, num_folds, device):
        self.num_folds = num_folds
        self.device = device
        
    # def _root_mean_squared_error(self, x, y):
    #     return torch.sqrt(torch.mean(torch.pow(x-y, 2.0)))
    
    # def _r2_score(self):
    #     pass
    
    # def _mean_absolute_error(self): 
    #     pass
        
    """
    Calculates regression metrics for given batch.
    """
    def eval_reg_batch_metrics(self, preds, targets):
        # rmse = self._root_mean_squared_error(preds, targets)
        if self.device != 'cpu':
            preds, targets = preds.cpu(), targets.cpu()
        preds, targets = preds.detach().numpy().flatten(), targets.detach().numpy().flatten()
        rmse = mean_squared_error(y_true=targets, y_pred=preds, squared=False)
        r2 = r2_score(y_true=targets, y_pred=preds)
        mae = mean_squared_error(y_true=targets, y_pred=preds)
        return {'rmse' : rmse, 'r2' : r2, 'mae' : mae}
    
    """
    Calculates classification metrics for given batch.
    """
    def eval_class_batch_metrics(self, preds, targets):
        pass
    
"""
Ideas:
1. Implement scores in Pytorch. 
"""
    
