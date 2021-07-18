#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 14:08:42 2021

@author: melike
"""

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import os
import os.path as osp
import torch
from torch import device
from torch.nn import MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader, Subset
from datasets import Lake2dDataset
from train import create_model, get_reg_min_max
from baseline import load_fold_sample_ids_args

HEAT_MAP_PARAMS = {'linewidth': 0,
                   'cmap': 'Spectral_r',
                   'xticklabels': False,
                   'yticklabels': False}

def generate_rand_heatmap():
    uniform_data = np.random.rand(600, 600)
    ax = sns.heatmap(uniform_data, **HEAT_MAP_PARAMS)
    plt.show()
    
"""
Takes an image id, calculates and returns its samples' indices in labeled and
unlabeled datasets. 
img_id, len_unlabeled_mask -> [int]
"""
def calc_image_sample_ids(img_id, len_labeled_dataset, len_unlabeled_dataset):
    labeled_ids = [i + img_id for i in range(0, len_labeled_dataset, 32)]
    unlabeled_ids = [i + img_id for i in range(0, len_unlabeled_dataset, 32)]
    return labeled_ids, unlabeled_ids

"""
Takes a model and subsets created from given image id, predicts each pixel's 
regression value. 
"""
def predict(model, loader, heatmap, reg_min, reg_max):
    model.eval()
    with torch.no_grad():
        for batch_id, data in enumerate(loader):
            patches, _, _, (img_ids, pxs, pys) = data
            patches = patches.to(args['device'])
            reg_preds, _ = model(patches)
            for j in range(len(reg_preds)):
                unnorm_reg = unnorm_reg_val(r=reg_preds[j], reg_min=reg_min, reg_max=reg_max)
                heatmap[pxs[j]][pys[j]] = unnorm_reg
            if batch_id % 100 == 0:
                print(batch_id)
    return heatmap                
    
"""
Takes an image, labeled and unlabeled datasets and model. Predicts image's 
regression values with the model and creates its heat map. 
"""
def generate_heatmap(img_id, labeled_dataset, unlabeled_dataset, model, reg_min, reg_max):
    labeled_ids, unlabeled_ids = calc_image_sample_ids(img_id=img_id,                                   # Retrieve image's sample ids
                                                       len_labeled_dataset=len(labeled_dataset),
                                                       len_unlabeled_dataset=len(unlabeled_dataset))
    labeled_subset = Subset(labeled_dataset, indices=labeled_ids)                                    # Load image's sample ids as subsets. 
    unlabeled_subset = Subset(unlabeled_dataset, indices=unlabeled_ids)
    
    labeled_loader = DataLoader(labeled_subset, **args['test'])                                         # Both can be loaded with test params since no shuffle is required. 
    unlabeled_loader = DataLoader(unlabeled_subset, **args['test'])
    
    heatmap = np.zeros([650, 650])
    for loader in [labeled_loader, unlabeled_loader]:
        heatmap = predict(model=model, loader=loader, heatmap=heatmap, 
                          reg_min=reg_min, reg_max=reg_max)
        
    ax = sns.heatmap(heatmap, **HEAT_MAP_PARAMS)
    plt.show()
    # train'den normalizasyon degerlerini al. 
    

# """
# Takes trained model's name and loads its args in order to creatie 
# datasets for prediction. 
# """
# def load_args(run_name):
#     run_path = osp.join(os.getcwd(), 'runs', run_name)
#     args_path = osp.join(run_path, 'args.txt')
#     if not osp.isfile(args_path):
#         raise Exception('Given args file could not be found on {}.'.format(args_path))
#     with open(args_path, 'rb') as f:                                            # Load args        
#         args = eval(f.read())
#     verify_args(args=args)
#     return args

"""
Verify args.
"""
def verify_args(args):
    if args['pred_type'] != 'reg+class':
        raise Exception('Expected pred_type to be reg+class. Given: {}'.format(args['pred_type']))
    if args['model'] != 'eaoriginaldan':
        raise Exception('Expected model to be EAOriginalDAN. Given: {}'.format(args['model']))
        
"""
Verify image id. 
"""
def verify_image_id(img_id):
    if img_id not in range(0, 10):
        raise Exception('Image id can be in [0, 9] range. Given: {}'.format(img_id))

"""
Takes a run name, args, model name and fold name, loads and returns the model. 
"""
def load_model(run_name, args, fold, model_name='best_test_score.pth'):
    model_path = osp.join(os.getcwd(), 'model_files', run_name, 'fold_{}'.format(fold), model_name)
    if not osp.isfile(model_path):
        raise Exception('Given model file could not be found on {}.'.format(model_path))
    model = create_model(args)                                             # Already places model to device. 
    model.load_state_dict(torch.load(model_path))
    return model

"""
Returns min, max regression values of given fold. 
"""
def get_fold_reg_minmax(fold, fold_sample_ids, labeled_dataset):
    tr_index = fold_sample_ids['tr_ids'][fold]
    reg_min, reg_max = get_reg_min_max(labeled_dataset.reg_vals[tr_index])
    return reg_min, reg_max

"""
Takes a (predicted) regression value, and min and max regression values from train set. 
Unnormalizes the given value. 
"""
def unnorm_reg_val(r, reg_min, reg_max):
    return (reg_max - reg_min) * (r + 1) / 2 + reg_min
    
if __name__ == "__main__":
    # generate_rand_heatmap()
    RUN_NAME = '2021_07_18__13_56_46'
    fold_sample_ids, args = load_fold_sample_ids_args(run_name=RUN_NAME)
    fold = 0
    
    """ Load model and datasets """
    model = load_model(run_name=RUN_NAME, args=args, fold=fold)
    dataset_params = {'patch_size': args['patch_size'],
                      'date_type': args['date_type'],
                      'reshape_to_mosaic': args['reshape_to_mosaic']}
    labeled_dataset = Lake2dDataset(learning='labeled', **dataset_params)
    unlabeled_dataset = Lake2dDataset(learning='unlabeled', **dataset_params)
    
    reg_min, reg_max = get_fold_reg_minmax(fold=fold,
                                       fold_sample_ids=fold_sample_ids, 
                                       labeled_dataset=labeled_dataset)
    print('reg values, min: {}, max: {}'.format(reg_min, reg_max))
    # for r in [-1.0, 0, 1.0]:
    #     unnorm_r = unnorm_reg_val(r=r, reg_min=reg_min, reg_max=reg_max)
    #     print('{} became {}'.format(r, unnorm_r))
    
    """ Generate heatmap for given image """
    img_id = 3
    verify_image_id(img_id=img_id)
    generate_heatmap(img_id=img_id, model=model,
                      labeled_dataset=labeled_dataset, 
                      unlabeled_dataset=unlabeled_dataset, 
                      reg_min=reg_min, reg_max=reg_max)
    
