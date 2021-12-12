#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 00:34:26 2021

@author: melike
"""

import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from os import path as osp
import constants as C
from datasets import Lake2dFoldDataset
from metrics import Metrics
from models import DandadaDAN, EANet, EADAN, EAOriginal, MultiLayerPerceptron, WaterNet, EAOriginalDAN, MDN, MaruMDN
from report import Report
from losses import AutomaticWeightedLoss
from baseline import load_data, load_args
from train import create_model, load_fold_sample_ids_args
from vis import save_estimates_targets, plot_estimates_targets

"""
Returns test set loader for given fold. Beware that it has to be normalized 
with labeled train set values, so that's why load_data() is used here. 
"""
def get_test_loader_args(run_name, best_run_name, best_fold):
    fold_sample_ids = load_fold_sample_ids_args(run_name=run_name)           # Sample ids come from this one.
    args = load_args(run_name=best_run_name)                                 # args come from best run.
    _, test_loader = load_data(args=args, fold=best_fold, 
                               fold_sample_ids=fold_sample_ids, return_loaders=True)
    return test_loader, args

"""
Loads best fold of the model given. 
"""
def load_model(model_name, best_fold, args):
    test_model = create_model(args)
    model_dir_path = osp.join(C.MODEL_DIR_PATH, args['run_name'], 'fold_' + str(best_fold))
    model_path = osp.join(model_dir_path, model_name)
    if osp.isfile(model_path):
        test_model.load_state_dict(torch.load(model_path))
        return test_model
    else:
        raise Exception('model: {} for fold#{} does not exist.'.format(model_name, best_fold))
    
"""
Returns estimated and target values
"""
def inference(run_name, best_run_name, model_name, best_fold):
    test_loader, args = get_test_loader_args(run_name, best_run_name, best_fold)
    model = load_model(model_name, best_fold, args)
    all_preds, all_targets = torch.tensor([]).to(args['device']), torch.tensor([]).to(args['device'])
    
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            patches, date_types, target_regs, (img_idxs, pxs, pys) = data
            patches, date_types, target_regs = patches.to(args['device']), date_types.to(args['device']), target_regs.to(args['device'])
            
            """ Estimation """
            if args['pred_type'] == 'reg':
                if args['model'] in ['mdn', 'marumdn']:                         # Regression with MDNs
                    pi, sigma, mu = model(patches)
                    reg_preds = MaruMDN.get_pred(pi_data=pi, sigma_data=sigma, mu_data=mu, n_samples=patches.shape[0])
                
                else:                                                           # Regression with DANs
                    reg_preds = model(patches)
                    if args['model'] in C.DAN_MODELS:
                        reg_preds, _ = reg_preds
                        
            elif args['pred_type'] == 'class':
                raise ValueError('This is for visualizing target and estimated regression values. So, it cannot be used for classification!')
                
            elif args['pred_type'] == 'reg+class':                              # Regression + classification with DANs
                reg_preds, _ = model(patches)
            
            all_preds = torch.cat([all_preds, reg_preds], dim=0)
            all_targets = torch.cat([all_targets, target_regs], dim=0)
            
    return all_preds, all_targets

if __name__ == "__main__":
    
    SAMPLE_IDS_FROM_RUN_NAME = '2021_07_01__11_23_50'
    best_run_name = '2021_07_07__23_02_22'
    model_name = 'best_test_score.pth'
    best_fold = 8
    
    preds, targets = inference(run_name=SAMPLE_IDS_FROM_RUN_NAME, best_run_name=best_run_name,
                               model_name=model_name, best_fold=best_fold)
    plot_estimates_targets(preds, targets, SAMPLE_IDS_FROM_RUN_NAME, model_name, best_fold, change_range=True)
    save_estimates_targets(preds, targets, SAMPLE_IDS_FROM_RUN_NAME, model_name, best_fold, change_range=True)

























