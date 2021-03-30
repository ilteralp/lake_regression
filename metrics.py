
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:40:07 2021

@author: melike
"""

import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, cohen_kappa_score, f1_score, accuracy_score
import numpy as np
import constants as C

class Metrics:
    """
    Measures regression and classification metrics.
    
    Args:
        num_folds (int): Number of folds. Can be None for training without 
        cross-validation or an int >0 for training with cross-validation. 
    """
    
    def __init__(self, num_folds, device, pred_type):
        self.num_folds = num_folds
        self.device = device
        self.pred_type = pred_type
        self.test_score_names = self._init_test_score_names()
        self.test_scores = self._init_test_scores()
        
    def _init_test_score_names(self):
        if self.pred_type == 'reg':
            return ['r2']
        elif self.pred_type == 'class':
            return ['kappa']
        elif self.pred_type == 'reg+class':
            return ['kappa', 'r2']
    
    """
    Create score dict to keep test scores of all folds.
    """
    def _init_test_scores(self):
        scores = {}
        for score_name in self.test_score_names:
            scores[score_name] = {'model_last_epoch.pth': [],                   # Last epoch model
                                  'best_val_loss.pth': [],                      # Best validation loss model
                                  'best_val_score.pth': []}                     # Best validation score model
        return scores
            
    """
    Adds one fold score to test scores. 
    """
    def add_fold_score(self, fold_score, model_name):
        for score_name in self.test_score_names:
            s = np.nanmean(fold_score[0][score_name])                           # Has result of each batch, so take mean of it. Kappa may return NaN, so nanmean is used. 
            self.test_scores[score_name][model_name].append(s)
            
    """
    Returns dict of mean and std for each model and score. 
    """
    def get_mean_std_test_results(self):
        result = {}
        for score_name in self.test_score_names:
            for model_name, all_fold_scores in self.test_scores[score_name].items():
                result[score_name][model_name]['mean'] = np.mean(all_fold_scores)
                result[score_name][model_name]['std'] = np.std(all_fold_scores)
        return result
                
            
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
        preds = torch.argmax(preds, 1)
        if self.device != 'cpu':
            preds, targets = preds.cpu(), targets.cpu()
        preds, targets = preds.detach().numpy().flatten(), targets.detach().numpy().flatten()
        kappa = cohen_kappa_score(preds, targets)
        f1 = f1_score(y_true=targets, y_pred=preds, average='macro')
        acc = accuracy_score(y_true=targets, y_pred=preds)
        return {'kappa' : kappa, 'f1' : f1, 'acc' : acc}
    
"""
Ideas:
1. Implement regression and classification scores in Pytorch. 
"""
    
