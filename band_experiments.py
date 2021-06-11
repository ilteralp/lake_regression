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

def estimate_model_a_clus(X_train):
    a, b = 80.7, 53.18
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    rat = rrs_705 / rrs_665
    return a * rat + b

def estimate_model_j_clus(X_train):
    a, b, c = 19.31, 153.5, 105.4
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    return a + b * rat + c * pow(rat, 2)

def estimate_model_j_cal(X_train):
    a, b, c = 18.44, 149.2, 374.9
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    return a + b * rat + c * pow(rat, 2)

def estimate_model_k_org(X_train):
    a, b, c = 14.039, 86.115, 194.33
    # a, b, c = 0.90780541, -43.16215739, 260.78290038
    rrs_665, rrs_705,  =  X_train[:, 3], X_train[:, 4]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    return a + b * rat + c * pow(rat, 2)
    
def estimate_model_a_clus2(X_train):
    a, b = 53.29, -30.08
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    return a * (rrs_705 / rrs_665) + b

# def estimate_model_n_clus(X_train):
#     a, b, c, d, e = 0.0536, 7.308, 116.2, 412.4, 463.5
#     rrs_490, rrs_560 = X_train[:, 1], X_train[:, 2]
#     rrs_560 = elim_zeros(rrs_560)
#     x = np.log10(rrs_490 / rrs_560)
#     x = x.astype('float64')
#     return pow(10, a + b * x + c * pow(x, 2) + d * pow(x, 3) + e * pow(x, 4))

# def estimate_model_m_clus2(X_train):
#     a, b, c, d, e = -5020, 2.9e+04, -6.1e+04, 5.749e+04, -2.026e+04
#     rrs_490, rrs_560 = X_train[:, 1], X_train[:, 2]
#     rrs_560 = elim_zeros(rrs_560)
#     x = np.log10(rrs_490 / rrs_560)
#     x = x.astype('float64')
#     return pow(10, a + b * x + c * pow(x, 2) + d * pow(x, 3) + e * pow(x, 4))

"""
Adds 1.0e-6 to zero values.  
"""
def elim_zeros(rrs):
    rrs[rrs == 0] = 1.0e-6
    return rrs

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
    # fs = [estimate_model_c_clus, estimate_model_a_clus, estimate_model_j_clus, estimate_model_j_cal, 
    #         estimate_model_k_org, estimate_model_a_clus2]
    # fnames = ['c_clus', 'a_clus', 'j_clus', 'j_cal', 'k_org', 'a_clus2']
    fs = [estimate_model_k_org]
    fnames = ['k_org']
    scores = { k: init_scores() for k in fnames}
    for fold in range(args['num_folds']):
        X_train, y_train, _, _ = load_data(args=args, fold=fold, fold_sample_ids=fold_sample_ids)       # There is no test set since a model is not trained. 
        if X_train.shape != (256, 12):
            raise Exception('Expected training set to be (256, 12). Given {}'.format(X_train.shape))
            
        for f, fname in zip(fs, fnames):
            y_pred = f(X_train=X_train)
            calc_scores(y_true=y_train, y_pred=y_pred, scores=scores[fname])
    
    for k, l in scores.items():
        print(k, ':')
        for s, v in l.items():
            print('{}, mean: {:.4f}, std: {:.4f}'.format(s, np.mean(v), np.std(v)))
    
"""
Band ratio models from Neil19. 
"""
def estimate_model_c_clus(X_train):
    a, b, c = 86.09, -517.5, 886.7
    # a, b, c = -2207.70406696, 5876.97668667, -3791.06652595
    # a, b, c = -44.48564812, 180.04721124, -139.61928711
    rrs_665, rrs_705,  =  X_train[:, 3], X_train[:, 4]
    rat = rrs_705 / rrs_665
    return a * pow(rat, 2) + b * rat + c

def solve_for_c_clus(X_train, y_train):
    num_vals = 3
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = rrs_705 / rrs_665
    xs = np.array([[rat[0] ** 2, rat[0], 1], [rat[1] ** 2, rat[1], 1], [rat[2] ** 2, rat[2], 1]])
    ys = np.asarray(regs)
    cons = np.linalg.solve(xs, ys)
    print(cons)
    
def solve_for_k_org(X_train, y_train):
    num_vals = 3
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    xs = np.array([[1, rat[0], rat[0] ** 2], [1, rat[1], rat[1] ** 2], [1, rat[2], rat[2] ** 2]])
    ys = np.asarray(regs)
    cons = np.linalg.solve(xs, ys)
    print('xs', xs)
    print('ys', ys)
    print('cons', cons)
    print('allclose:', np.allclose(np.dot(xs, cons), ys))
    return xs, ys, cons
    
    
if __name__ == "__main__":
    run_name = '2021_05_29__23_59_42'
    estimate_on_folds(run_name)
    # fold_sample_ids, args = load_fold_sample_ids_args(run_name=run_name)
    # args['patch_norm'], args['reg_norm'] = False, True                                                 # Don't normalize image and Chl-a values. 
    # args['patch_size'] = 1
    # X_train, y_train, _, _ = load_data(args=args, fold=0, fold_sample_ids=fold_sample_ids)
    # # solve_for_c_clus(X_train, y_train)
    # xs, ys, cons = solve_for_k_org(X_train, y_train)
    
