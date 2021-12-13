#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 15:55:32 2021

@author: melike
"""

import os
from os import path as osp
import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, random_split, Subset, DataLoader
from os import path as osp
import constants as C
from datasets import Lake2dFoldDataset, Lake2dDataset
from metrics import Metrics
from models import DandadaDAN, EANet, EADAN, EAOriginal, MultiLayerPerceptron, WaterNet, EAOriginalDAN, MDN, MaruMDN
from losses import AutomaticWeightedLoss
from train import _test, create_model
from inference import load_model, get_test_loader_args
from baseline import load_data, load_args


"""
Returns unlabeled datasets. 
"""
def get_unlabeled_dataset_loader(args):
    unlabeled_set = Lake2dDataset(learning='unlabeled', date_type=args['date_type'], 
                              patch_size=args['patch_size'], 
                              reshape_to_mosaic=args['reshape_to_mosaic'])
    unlabeled_loader = DataLoader(unlabeled_set, **args['unlabeled'])
    return unlabeled_loader

"""
Returns min/max of confusion matrices. 
"""
def _get_vmin_vmax(conf_mats):
    vmin, vmax = np.min(conf_mats[0]), np.max(conf_mats[0])           # Get min, max for visualizations. 
    for cm in conf_mats:
        _min, _max = np.min(cm), np.max(cm)
        vmin = _min if _min < vmin else vmin
        vmax = _max if _max > vmax else vmax
    return vmin, vmax

"""
Takes a list of confusion matrices and visualizes them. 
"""
def visualize_conf_mats(conf_mats, names, best_fold):
    fig, axn = plt.subplots(1, len(conf_mats), figsize=(16, 7))
    cbar_ax = fig.add_axes([.91, .3, .03, .4])
    vmin, vmax = _get_vmin_vmax(conf_mats)
    
    for i, ax in enumerate(axn.flat):
        df_cm = pd.DataFrame(conf_mats[i], index = [i for i in C.MONTHS],
                             columns = [i for i in C.MONTHS])
        annot = np.diag(np.diag(conf_mats[i]))
        annot = np.round(annot, 2)
        annot = annot.astype('str')
        annot[annot == '0.0'] = ''
        ax.title.set_text(names[i])
        im = sn.heatmap(df_cm, ax=ax, annot=annot, fmt='', square=True, vmin=vmin, vmax=vmax,
                cbar=i == 0, cbar_ax=None if i else cbar_ax, cmap="PRGn")                     # Draw cbar only for the first iter.
        
    str_names = '_'.join(map(str, names))
    plt.savefig(osp.join(os.getcwd(), 'vis', 'conf_mats_names={}_fold={}.png'.format(str_names, best_fold).format(str_names, best_fold)), 
                format='png', dpi=300, bbox_inches='tight')


"""
Generates confusion matrix for given fold. 
"""
def generate_conf_mat(run_name, best_run_name, model_name, best_fold):
    # test_loader, args = get_test_loader_args(run_name, best_run_name, best_fold, unlabeled=True) # Load unlabeled samples, because conf.mat is filled with 1's with labeled one. 
    args = load_args(best_run_name)
    test_loader = get_unlabeled_dataset_loader(args)
    model = load_model(model_name, best_fold, args)
    # model = create_model(args)                                                  # Testing with untrained model. 
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
            
            if batch_id % 10 == 0:
                print('{} of {}'.format(batch_id, len(test_loader)))
           
    # print('conf mat')
    # print(metrics.conf_mat[best_fold])
    print('Normed conf mat')
    print(metrics.get_normed_conf_mat()[best_fold])

if __name__ == "__main__":
    SAMPLE_IDS_FROM_RUN_NAME = '2021_07_01__11_23_50'
    # best_run_name = '2021_07_07__23_02_22'                                      # best MTL model (trained on 2.3M unlabeled samples)
    best_run_name = '2021_07_05__00_09_49'                                      # MTL model trained with 503K unlabeled samples
    # best_run_name = '2021_07_04__20_02_39'                                      # MTL model trained with 28K unlabeled samples
    # best_run_name = '2021_12_07__17_18_27'                                      # Classification with labeled samples only

    model_name = 'best_test_score.pth'
    best_fold = 8
    
    generate_conf_mat(run_name=SAMPLE_IDS_FROM_RUN_NAME, best_run_name=best_run_name,
                                model_name=model_name, best_fold=best_fold)
    
    # num_classes = 12
    # # conf_mats = [np.random.rand(num_classes, num_classes) for i in range(2)]
    # # names = ['unlabeled 28K samples', 'unlabeled 2.3M samples']
    # names = ['Reg+Class, 2.3M unlabeled (2021_07_07__23_02_22)', 'Classification only (2021_12_07__17_18_27)']
    
    # """ """
    # _best_1 = torch.tensor([[9.5274e-01, 3.8131e-02, 1.2516e-03, 5.1418e-04, 7.4421e-05, 8.7952e-05,
    #      4.7359e-05, 1.4411e-03, 5.4124e-04, 3.6534e-03, 1.2246e-03, 2.9768e-04],
    #     [2.3003e-03, 9.7367e-01, 2.9137e-03, 4.2848e-03, 6.8557e-04, 3.0670e-04,
    #      1.9395e-04, 4.6141e-03, 1.6733e-03, 7.0813e-04, 7.8345e-03, 8.1186e-04],
    #     [5.3222e-04, 8.5742e-03, 9.8205e-01, 2.5438e-03, 2.3003e-03, 1.8492e-04,
    #      1.2629e-04, 8.0735e-04, 1.8041e-04, 5.1418e-04, 3.6083e-04, 1.8222e-03],
    #     [9.9228e-05, 4.1585e-03, 1.7906e-03, 9.8213e-01, 1.8177e-03, 2.6205e-03,
    #      4.2488e-03, 2.2552e-05, 0.0000e+00, 1.7951e-03, 1.3170e-03, 0.0000e+00],
    #     [4.0593e-05, 8.8403e-04, 3.6083e-05, 5.5026e-04, 9.9558e-01, 1.8898e-03,
    #      3.2926e-04, 0.0000e+00, 0.0000e+00, 4.2397e-04, 2.6160e-04, 0.0000e+00],
    #     [0.0000e+00, 3.8564e-04, 8.7952e-05, 7.4083e-04, 6.5490e-03, 9.3141e-01,
    #      5.9516e-02, 0.0000e+00, 6.9347e-04, 5.6154e-04, 5.7507e-05, 0.0000e+00],
    #     [9.0207e-06, 4.4201e-04, 1.5335e-04, 2.5709e-03, 6.6753e-03, 2.5727e-02,
    #      9.6248e-01, 1.8492e-04, 1.3576e-03, 3.3828e-04, 6.3145e-05, 0.0000e+00],
    #     [5.8860e-04, 2.8009e-02, 1.2178e-04, 2.1650e-04, 0.0000e+00, 2.7062e-05,
    #      1.3531e-05, 9.6267e-01, 3.4098e-03, 1.2178e-03, 3.4166e-03, 3.0445e-04],
    #     [1.5764e-03, 2.6210e-02, 1.9620e-04, 1.0148e-04, 1.7590e-04, 2.4356e-04,
    #      1.7793e-03, 5.8725e-03, 9.6052e-01, 1.1704e-03, 1.8199e-03, 3.3828e-04],
    #     [4.6547e-03, 2.1311e-04, 1.8943e-04, 2.7739e-04, 4.2285e-04, 1.2482e-03,
    #      4.3976e-05, 1.4884e-04, 6.4273e-05, 9.9249e-01, 2.4694e-04, 0.0000e+00],
    #     [1.4749e-03, 6.0491e-02, 1.1975e-03, 7.2053e-03, 1.7049e-03, 6.6302e-04,
    #      4.4653e-04, 9.8236e-03, 2.7130e-03, 2.3003e-03, 9.1097e-01, 1.0148e-03],
    #     [1.6237e-03, 8.0104e-02, 6.1025e-03, 1.7590e-04, 4.0593e-05, 1.3531e-04,
    #      1.3531e-05, 4.6817e-03, 1.6779e-03, 1.2178e-04, 1.0690e-03, 9.0425e-01]])
    
    # """ """
    # _best_2 = torch.
    
    # """ """
    # _best_3 = torch.
    
    # """ """
    # _best_4 = torch.tensor([[8.7189e-01, 1.0162e-02, 3.6534e-04, 4.4653e-04, 3.5857e-04, 2.4356e-04,
    #      2.9092e-04, 3.5316e-03, 2.5980e-03, 1.0951e-01, 5.1418e-04, 8.7952e-05],
    #     [3.0806e-03, 8.2469e-01, 7.9833e-03, 2.3747e-02, 1.2011e-02, 1.6914e-03,
    #      2.1650e-03, 9.4492e-03, 7.9112e-03, 8.5232e-02, 2.1375e-02, 6.6302e-04],
    #     [4.1044e-04, 1.2349e-02, 9.1502e-01, 8.2900e-03, 4.1585e-02, 8.5246e-04,
    #      4.4607e-03, 1.2088e-03, 2.1424e-03, 1.0640e-02, 1.0690e-03, 1.9665e-03],
    #     [3.2475e-04, 1.5651e-03, 0.0000e+00, 8.3424e-01, 1.1939e-02, 1.9467e-02,
    #      9.7469e-02, 0.0000e+00, 0.0000e+00, 3.4991e-02, 0.0000e+00, 0.0000e+00],
    #     [1.3531e-05, 9.0207e-06, 0.0000e+00, 8.1186e-05, 9.8155e-01, 1.6594e-02,
    #      2.6160e-04, 0.0000e+00, 0.0000e+00, 1.4884e-03, 0.0000e+00, 0.0000e+00],
    #     [3.3828e-06, 0.0000e+00, 3.3828e-06, 3.9308e-03, 1.5635e-02, 9.2317e-01,
    #      5.3996e-02, 0.0000e+00, 1.5865e-03, 1.6745e-03, 0.0000e+00, 0.0000e+00],
    #     [0.0000e+00, 4.9614e-05, 0.0000e+00, 6.8106e-04, 1.4519e-02, 2.3321e-01,
    #      7.4706e-01, 9.0207e-06, 5.9988e-04, 3.8699e-03, 0.0000e+00, 0.0000e+00],
    #     [3.3151e-04, 6.8129e-03, 0.0000e+00, 8.7952e-05, 1.3531e-05, 4.0593e-05,
    #      2.1650e-04, 9.3893e-01, 4.2555e-03, 4.6892e-02, 2.4085e-03, 6.7655e-06],
    #     [1.2516e-03, 6.4070e-03, 0.0000e+00, 5.4124e-05, 1.3531e-05, 2.0297e-05,
    #      4.9591e-03, 3.1013e-02, 9.1170e-01, 4.3455e-02, 1.1163e-03, 1.3531e-05],
    #     [1.7712e-02, 0.0000e+00, 1.0148e-04, 1.8741e-03, 3.0749e-03, 1.0994e-03,
    #      2.0297e-04, 1.0148e-05, 2.5371e-04, 9.7567e-01, 0.0000e+00, 0.0000e+00],
    #     [2.1041e-03, 7.0592e-02, 1.8673e-03, 9.5056e-03, 1.5655e-02, 1.5831e-03,
    #      4.4450e-03, 3.9531e-02, 6.9888e-03, 5.0829e-02, 7.9641e-01, 4.8712e-04],
    #     [1.1231e-03, 1.5791e-02, 2.0838e-03, 2.1650e-04, 1.5290e-03, 2.0297e-04,
    #      1.9891e-03, 3.1121e-03, 7.8345e-03, 9.6490e-02, 1.9349e-03, 8.6769e-01]])
    # """ """
    # conf_mats = [_best_1.numpy(), _best_4.numpy()]
    # visualize_conf_mats(conf_mats, names, best_fold)

    