#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 00:40:59 2021

@author: melike
"""
from sklearn.svm import SVR
import numpy as np
import os
import os.path as osp
import pickle
import torch
from torch.nn import MSELoss, CrossEntropyLoss
from torch import device
from datasets import Lake2dDataset, Lake2dFoldDataset
from torch.utils.data import Subset, DataLoader
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import LinearSVR
from train import calc_mean_std, load_reg_min_max, get_reg_min_max

def basic_svr():
    # n_samples, n_features = 10, 5
    # rng = np.random.RandomState(0)
    # y = rng.randn(n_samples)
    # X = rng.randn(n_samples, n_features)
    X = np.sort(5 * np.random.rand(40, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(8))                                     # Add noise

    # for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    #     regressor = SVR(kernel=kernel)
    #     regressor.fit(X, y)
    #     print(f'kernel: {kernel}, R2: {regressor.score(X, y):.4f}')
    
    
    reg = LinearSVR(max_iter=3000, tol=1e-3, epsilon=0.1)
    reg.fit(X, y)
    print(f'LinearSVR, R2: {reg.score(X, y):.4f}')
        
"""
Loads and return fold sample ids from given path. 
"""
def load_fold_sample_ids_args(run_name):
    run_path = osp.join(os.getcwd(), 'runs', run_name)
    args_path, sample_ids_path = osp.join(run_path, 'args.txt'), osp.join(run_path, 'sample_ids.pickle')
    if not osp.isfile(args_path) or not osp.isfile(sample_ids_path):
        raise Exception('Given pickle file or sample ids file could not be found on {}.'.format(run_path))
        
    with open(sample_ids_path, 'rb') as f:                                      # Load sample ids    
        fold_sample_ids = pickle.load(f)
    with open(args_path, 'rb') as f:                                            # Load args        
        args = eval(f.read())
    return fold_sample_ids, args
        
"""
Takes args and fold number. Loads and returns that fold's train and test samples
in X, y format. 
"""
def load_data(args, fold, fold_sample_ids):
    # Create dataset
    dataset_dict = {'learning': 'labeled',
                    'date_type': args['date_type'],
                    'patch_size': args['patch_size'],
                    'reshape_to_mosaic': False }

    tr_ids = fold_sample_ids['tr_ids'][fold]
    test_ids = fold_sample_ids['test_ids'][fold]
                        
    if args['fold_setup'] == 'random':
        dataset = Lake2dDataset(**dataset_dict)
        tr_set = Subset(dataset, indices=tr_ids)
        test_set = Subset(dataset, indices=test_ids)
    else:                                                                       # Spatial, temporal_day, temporal_year
        tr_set = Lake2dFoldDataset(**dataset_dict, fold_setup=args['fold_setup'],
                                   ids=tr_ids)
        test_set = Lake2dFoldDataset(**dataset_dict, fold_setup=args['fold_setup'],
                                     ids=test_ids)
    """ Normalize patches """
    if args['patch_norm']:
        patches_mean, patches_std = calc_mean_std(DataLoader(tr_set, **args['tr']))
        if args['fold_setup'] == 'random':
            dataset.set_patch_mean_std(means=patches_mean, stds=patches_std)
        else:
            for d in [tr_set, test_set]:
                d.set_patch_mean_std(means=patches_mean, stds=patches_std)
                
    """ Normalize regression values """
    if args['reg_norm']:
        reg_min, reg_max = load_reg_min_max(DataLoader(tr_set, **args['tr']))
        if args['fold_setup'] == 'random':
            reg_min, reg_max = get_reg_min_max(dataset.reg_vals[tr_ids])
            dataset.set_reg_min_max(reg_min=reg_min, reg_max=reg_max)
        else:
            for d in [tr_set, test_set]:
                d.set_reg_min_max(reg_min=reg_min, reg_max=reg_max)
    
    """ Load data with laoders """
    tr_loader = DataLoader(tr_set, **args['tr'])
    test_loader = DataLoader(test_set, **args['test'])
    
    X_train, y_train, X_test, y_test = [], [], [], []
    for data in tr_loader:
        patch, date_type, reg_val, (img_idx, px, py) = data
        X_train.append(patch)
        y_train.append(reg_val)
    
    for data in test_loader:
        patch, date_type, reg_val, (img_idx, px, py) = data
        X_test.append(patch)
        y_test.append(reg_val)
        
    datasets = [torch.cat(d) for d in [X_train, y_train, X_test, y_test]]
    datasets = [d.reshape(d.shape[0], -1).numpy() if d.dim() == 4 else d.squeeze().numpy() for d in datasets]
    return datasets

"""
Takes train set and labels, fits the model and returns it.  
"""
def train(X_train, y_train, params):
    # regressor = SVR(kernel='linear', cache_size=7000)
    # regressor = LinearSVR(max_iter=50000, dual=False, loss='squared_epsilon_insensitive',
    #                       tol=1e-3, epsilon=0.1)
    regressor = LinearSVR(**params)
    regressor.fit(X_train, y_train)
    return regressor

"""
Calculates regression scores for given fold
"""
def calc_scores(regressor, X_test, y_test, scores):
    r2 = regressor.score(X_test, y_test)
    r = np.sqrt(r2) if r2 > 0 else -np.sqrt(-r2)
    y_pred = regressor.predict(X_test)
    rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    for name, s in zip(['r2', 'r', 'rmse', 'mae'], [r2, r, rmse, mae]):
        scores[name].append(s)
    
"""
Load data and train on folds.
"""
def train_on_folds(run_name):
    fold_sample_ids, args = load_fold_sample_ids_args(run_name=run_name)
    grid = {'dual': False, 'loss': 'epsilon_insensitive'}
    
    for c in [10. ** np.arange(-3, 3)]:
        grid['c'] = c
        scores = {'r2': [], 'r': [], 'mae': [], 'rmse': []}
        for fold in range(args['num_folds']):
            X_train, y_train, X_test, y_test = load_data(args=args, fold=fold, fold_sample_ids=fold_sample_ids)
            regressor = train(X_train=X_train, y_train=y_train, params=grid)
            calc_scores(regressor=regressor, X_test=X_test, y_test=y_test, scores=scores)
        
        print('grid: {}'.format(grid))
        for k, v in scores.items():
            print('{}, mean: {:.4f}, std: {:.4f}'.format(k, np.mean(v), np.std(v)))

if __name__ == "__main__":
    # basic_svr()
    
    RUN_NAME = '2021_07_01__11_23_50'
    train_on_folds(run_name=RUN_NAME)
    
    # ARGS_PATH = '/home/melike/repos/lake_regression/runs/2021_05_29__00_07_24/args.txt'
    
    
    
    
    
    
    
    
    