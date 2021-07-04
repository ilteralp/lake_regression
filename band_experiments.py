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
import itertools
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, r2_score
from baseline import load_data, load_fold_sample_ids_args

""" ================ Band ratio models from Neil19. =============== """
def estimate_model_a_clus(X_train, coeffs=None):
    a, b = (80.7, 53.18) if coeffs is None else coeffs
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    rat = rrs_705 / rrs_665
    return a * rat + b

def estimate_model_j_clus(X_train, coeffs=None):
    a, b, c = (19.31, 153.5, 105.4) if coeffs is None else coeffs
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    return a + b * rat + c * pow(rat, 2)

def estimate_model_j_cal(X_train, coeffs=None):
    a, b, c = (18.44, 149.2, 374.9) if coeffs is None else coeffs
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    return a + b * rat + c * pow(rat, 2)

def estimate_model_a_clus2(X_train, coeffs=None):
    a, b = (53.29, -30.08) if coeffs is None else coeffs
    rrs_665, rrs_705 = X_train[:, 3], X_train[:, 4]
    return a * (rrs_705 / rrs_665) + b

def estimate_model_k_org(X_train, coeffs=None):
    a, b, c = (14.039, 86.115, 194.33) if coeffs is None else coeffs
    rrs_665, rrs_705,  =  X_train[:, 3], X_train[:, 4]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    return a + b * rat + c * pow(rat, 2)
    
def estimate_model_c_clus(X_train, coeffs=None):
    a, b, c = (86.09, -517.5, 886.7) if coeffs is None else coeffs
    rrs_665, rrs_705,  =  X_train[:, 3], X_train[:, 4]
    rat = rrs_705 / rrs_665
    return a * pow(rat, 2) + b * rat + c

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
Adds 1.0e-6 to zero values due to division by zero.  
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
Estimate Chl-a values on folds
"""
def estimate_on_folds(run_name):
    fold_sample_ids, args = load_samples_set_args(run_name=run_name)
    # fs = [estimate_model_c_clus, estimate_model_a_clus, estimate_model_j_clus, estimate_model_j_cal, 
    #         estimate_model_k_org, estimate_model_a_clus2]
    # fnames = ['c_clus', 'a_clus', 'j_clus', 'j_cal', 'k_org', 'a_clus2']
    fs = [estimate_model_c_clus]
    fnames = ['c_clus']
    scores = { k: init_scores() for k in fnames}
    for fold in range(args['num_folds']):
        X_train, y_train, _, _ = load_data(args=args, fold=fold, fold_sample_ids=fold_sample_ids)       # There is no test set since a model is not trained. 
        if X_train.shape != (288, 12):
            raise Exception('Expected training set to be (288, 12). Given {}'.format(X_train.shape))
            
        for f, fname in zip(fs, fnames):
            y_pred = f(X_train=X_train)
            calc_scores(y_true=y_train, y_pred=y_pred, scores=scores[fname])
    
    for k, l in scores.items():
        print(k, ':')
        for s, v in l.items():
            print('{}, mean: {:.4f}, std: {:.4f}'.format(s, np.mean(v), np.std(v)))
            
            
"""
Estimate Chl-a value on a band-based model.
"""
def estimate_model_on_folds(fold_sample_ids, model, name, args, coeffs):
    scores = init_scores()
    for fold in range(args['num_folds']): 
        X_train, y_train, _, _ = load_data(args=args, fold=fold, fold_sample_ids=fold_sample_ids)
        if X_train.shape != (288, 12):
            raise Exception('Expected training set to be (288, 12). Given {}'.format(X_train.shape))
        
        y_pred = model(X_train=X_train, coeffs=coeffs)
        calc_scores(y_true=y_train, y_pred=y_pred, scores=scores)
        
    print('model: {} with coeffs: {}'.format(name, coeffs))
    for s, v in scores.items():
        print('{}, mean: {:.4f}, std: {:.4f}'.format(s, np.mean(v), np.std(v)))
            
    
""" ========= Find calibration coefficients of each model ========= """


def solve_for_c_clus(X_train, y_train):
    num_vals = 3
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = rrs_705 / rrs_665
    xs = np.array([[rat[0] ** 2, rat[0], 1], [rat[1] ** 2, rat[1], 1], [rat[2] ** 2, rat[2], 1]])
    ys = np.asarray(regs)
    coeffs = np.linalg.solve(xs, ys)
    print('c_clus, coeffs: {}, all_close: {}'.format(coeffs, np.allclose(np.dot(xs, coeffs), ys)))
    return coeffs[0], coeffs[1], coeffs[2]
    
def solve_for_k_org(X_train, y_train):
    num_vals = 3
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    xs = np.array([[1, rat[0], rat[0] ** 2], [1, rat[1], rat[1] ** 2], [1, rat[2], rat[2] ** 2]])
    ys = np.asarray(regs)
    coeffs = np.linalg.solve(xs, ys)
    print('k_org, coeffs: {}, all_close: {}'.format(coeffs, np.allclose(np.dot(xs, coeffs), ys)))
    return coeffs[0], coeffs[1], coeffs[2]

def solve_for_a_clus2(X_train, y_train):
    num_vals = 2
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = (rrs_705 / rrs_665)
    xs = np.array([[rat[0], 1], [rat[1], 1]])
    ys = np.asarray(regs)
    coeffs = np.linalg.solve(xs, ys)
    print('a_clus2, coeffs: {}, all_close: {}'.format(coeffs, np.allclose(np.dot(xs, coeffs), ys)))
    return coeffs[0], coeffs[1]

def solve_for_j_cal(X_train, y_train):
    num_vals = 3
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    xs = [[1, rat[0], rat[0] ** 2], [1, rat[1], rat[1] ** 2], [1, rat[2], rat[2] ** 2]]
    ys = np.asarray(regs)
    coeffs = np.linalg.solve(xs, ys)
    print('j_cal, coeffs: {}, all_close: {}'.format(coeffs, np.allclose(np.dot(xs, coeffs), ys)))
    return coeffs[0], coeffs[1], coeffs[2]

def solve_for_j_clus(X_train, y_train):
    num_vals = 3
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = (rrs_705 - rrs_665) / (rrs_705 + rrs_665)
    xs = [[1, rat[0], rat[0] ** 2], [1, rat[1], rat[1] ** 2], [1, rat[2], rat[2] ** 2]]
    ys = np.asarray(regs)
    coeffs = np.linalg.solve(xs, ys)
    print('j_clus, coeffs: {}, all_close: {}'.format(coeffs, np.allclose(np.dot(xs, coeffs), ys)))
    return coeffs[0], coeffs[1], coeffs[2]

def solve_for_a_clus(X_train, y_train):
    num_vals = 2
    rrs_665, rrs_705, regs = X_train[0:num_vals, 3], X_train[0:num_vals, 4], y_train[0:num_vals]
    rat = (rrs_705 / rrs_665)
    xs = np.array([[rat[0], 1], [rat[1], 1]])
    ys = np.asarray(regs)
    coeffs = np.linalg.solve(xs, ys)
    print('a_clus, coeffs: {}, all_close: {}'.format(coeffs, np.allclose(np.dot(xs, coeffs), ys)))
    return coeffs[0], coeffs[1]

def load_samples_set_args(run_name):
    fold_sample_ids, args = load_fold_sample_ids_args(run_name=run_name)
    args['patch_norm'], args['reg_norm'] = False, False                                             # Don't normalize image and Chl-a values. 
    args['patch_size'] = 1
    return fold_sample_ids, args
    
if __name__ == "__main__":
    run_name = '2021_06_18__16_08_05'
    
    fold_sample_ids, args = load_samples_set_args(run_name=run_name)
    X_train, y_train, _, _ = load_data(args=args, fold=0, fold_sample_ids=fold_sample_ids)
    models = {'c_clus': {'solver': solve_for_c_clus,
                          'model': estimate_model_c_clus},
              'k_org': {'solver': solve_for_k_org,
                        'model': estimate_model_k_org},
              'a_clus2': {'solver': solve_for_a_clus2,
                          'model': estimate_model_a_clus2},
              'j_cal': {'solver': solve_for_j_cal,
                          'model': estimate_model_j_cal},
              'j_clus': {'solver': solve_for_j_clus,
                          'model': estimate_model_j_clus},
              'a_clus': {'solver': solve_for_a_clus,
                         'model': estimate_model_a_clus}}
    patch_norms = [False]
    reg_norms = [True, False]
    
    
    for (patch_norm, reg_norm) in itertools.product(patch_norms, reg_norms):
        args['patch_norm'] = patch_norm
        args['reg_norm'] = reg_norm

        for fname, v in models.items():
            print('patch norm: {}, reg norm: {}'.format(args['patch_norm'], args['reg_norm']))
            coeffs = v['solver'](X_train, y_train)                                               # Calculate coefficients
            for x in [coeffs, None]:                                                             # Calculate regression scores.
                estimate_model_on_folds(fold_sample_ids=fold_sample_ids, model=v['model'], 
                                        name=fname, args=args, coeffs=x)
                print('=' * 72)

    
