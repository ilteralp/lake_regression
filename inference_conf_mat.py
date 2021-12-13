#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:55:32 2021

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
from losses import AutomaticWeightedLoss
from baseline import load_data, load_args
from train import _test
from inference import load_model, get_test_loader_args


"""
Generates confusion matrix for given fold. 
"""
def generate_conf_mat(run_name, best_run_name, model_name, best_fold):
    test_loader, args = get_test_loader_args(run_name, best_run_name, best_fold)
    model = load_model(model_name, best_fold, args)
    metrics = Metrics(num_folds=args['num_folds'], device=args['device'].type, 
                      pred_type=args['pred_type'], num_classes=args['num_classes'],
                      set_name='test')
    
    """ Predict labels """
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(test_loader):
            patches, date_types, target_regs, (img_idxs, pxs, pys) = data
            patches, date_types, target_regs = patches.to(args['device']), date_types.to(args['device']), target_regs.to(args['device'])
            
            if args['pred_type'] == 'reg':
                raise ValueError('Confusion matrix can be calculated for classification or reg+class. Given regression!')
            
            elif args['pred_type'] == 'class':
                class_preds = model(patches)
                if args['model'] in C.DAN_MODELS:
                    _, class_preds = class_preds
                    
            elif args['pred_type'] == 'reg+class':
                _, class_preds = model(patches)
                
            metrics.update_conf_matrix(preds=class_preds, targets=date_types, fold=best_fold) # Update confusion matrix
           
    print('conf mat')
    print(metrics.conf_mat[best_fold])
    print('Normed conf mat')
    print(metrics.get_normed_conf_mat()[best_fold])

if __name__ == "__main__":
    SAMPLE_IDS_FROM_RUN_NAME = '2021_07_01__11_23_50'
    # best_run_name = '2021_07_07__23_02_22'                                      # best MTL model (2.3M unlabeled samples)
    best_run_name = '2021_07_04__20_02_39'                                        # MTL model with 28K unlabeled samples

    model_name = 'best_test_score.pth'
    best_fold = 8
    
    generate_conf_mat(run_name=SAMPLE_IDS_FROM_RUN_NAME, best_run_name=best_run_name,
                               model_name=model_name, best_fold=best_fold)
    