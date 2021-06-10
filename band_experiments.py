#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:39:22 2021

@author: melike
"""

import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch import device
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, r2_score
from baseline import load_data, load_fold_sample_ids_args

"""
Estimate with model A from Neil19.
"""
def estimate_model_a(X_train):
    pass

"""
Estimate with model K from Neil19.
"""
def estimate_model_k(X_train):
    a, b, c = 14.039, 86.115, 194.325
    rrs_665, rrs_705,  =  X_train[:, 3], X_train[:, 4]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    return a + b * rat + c * pow(rat, 2)

"""
Estimate with model L from Neil19.
"""
def estimate_model_l(X_train):
    pass
    # a, b, c, d, e = 0.3255, -2.7677, 2.4409, -1.1288, -0.4990
    # rrs_443, rrs_490, rrs_510, = X_train[:, 0], X_train[:, 1]
    # Not applicable, we don't have a band close to rrs_510. 

"""
Band ratio model from Neil19. 
"""
def estimate_model_c_clus(X_train):
    a, b, c = 86.09, -517.5, 886.7
    rrs_665, rrs_705,  =  X_train[:, 3], X_train[:, 4]
    return a * pow(rrs_705 / rrs_665, 2) + b * (rrs_705 / rrs_665) + c

"""
Calculate regression scores. 
"""
def calc_scores(y_true, y_pred, scores):
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    r = np.sqrt(r2) if r2 > 0 else -np.sqrt(-r2)
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    for name, s in zip(['r2', 'r', 'rmse', 'mae'], [r2, r, rmse, mae]):
        scores[name].append(s)
        
def init_scores():
    return {'r2': [], 'r': [], 'mae': [], 'rmse': []}

"""
Estimate Chl-a valeus on folds
"""
def estimate_on_folds(run_name):
    fold_sample_ids, args = load_fold_sample_ids_args(run_name=run_name)
    args['patch_norm'], args['reg_norm'] = False, False                                                 # Don't normalize image and Chl-a values. 
    args['patch_size'] = 1                                                                              # Patch size is 1 since only pixels are used. 
    scores_c_clus, scores_model_k = init_scores(), init_scores()
    for fold in range(args['num_folds']):
        X_train, y_train, _, _ = load_data(args=args, fold=fold, fold_sample_ids=fold_sample_ids)       # There is no test set since a model is not trained. 
        if X_train.shape != (256, 12):
            raise Exception('Expected training set to be (256, 12). Given {}'.format(X_train.shape))
            
        y_pred_c_clus = estimate_model_c_clus(X_train=X_train)                                          # Estimate with band ratio algorithm. 
        y_pred_model_k = estimate_model_k(X_train=X_train)
        calc_scores(y_true=y_train, y_pred=y_pred_c_clus, scores=scores_c_clus)
        calc_scores(y_true=y_train, y_pred=y_pred_model_k, scores=scores_model_k)

    print('Model c_clus:')
    for k, v in scores_c_clus.items():
        print('{}, mean: {:.4f}, std: {:.4f}'.format(k, np.mean(v), np.std(v)))
    print('\nModel K:')
    for k, v in scores_model_k.items():
        print('{}, mean: {:.4f}, std: {:.4f}'.format(k, np.mean(v), np.std(v)))
    
    
if __name__ == "__main__":
    run_name = '2021_05_29__23_59_42'
    estimate_on_folds(run_name)
    
