#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 21:41:01 2021

@author: melike
"""

import os
from os import path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import itertools
from models import EAOriginalDAN
import matplotlib.pyplot as plt
from inference import load_model, get_test_loader_args


"""
Returns a sample from test set and its name. 
"""
def get_test_sample_args(run_name, best_run_name, best_fold):
    test_loader, args = get_test_loader_args(run_name, best_run_name, best_fold)
    test_iter = iter(test_loader)
    data = next(test_iter)
    patches, reg_vals, date_types, (img_ids, pxs, pys) = data
    print('patches[0].shape: {}'.format(patches[0].shape))
    return {'patch': patches[0], 'reg_val': reg_vals[0], 'img_id': img_ids[0]}, args

"""
Returns min/max of given feature maps. Will be used for visualization
"""
def get_min_max(activation):
    reg_min, class_min = torch.min(activation['reg_tanh']), torch.min(activation['class_tanh'])
    reg_max, class_max = torch.max(activation['reg_tanh']), torch.max(activation['class_tanh'])
    _min = reg_min if reg_min < class_min else class_min
    _max = reg_max if reg_max > class_max else class_max
    return _min, _max

"""
Used for hooking
"""
def get_activation(activation, name, task_name):
    def hook(model, input, output):
        activation[task_name + '_' + name] = output.detach()
    return hook

"""
Generates features maps of each task and saves them in dict. 
"""
def generate_feature_maps(model_name, best_fold, test_sample, args):
    model = load_model(model_name, best_fold, args)
    
    activation = {}
    model.regressor[0][2].register_forward_hook(get_activation(activation=activation, name='tanh', task_name='reg'))
    model.classifier[0][2].register_forward_hook(get_activation(activation=activation, name='tanh', task_name='class'))
    
    _ = model(test_sample.unsqueeze(0))                                          # Feed model one test sample to see its activations.
    activation['reg_tanh'] = activation['reg_tanh'].view(16, -1)                 # Change view, otherwise they (128, 1) which is hard to comprehend visually.
    activation['class_tanh'] = activation['class_tanh'].view(16, -1)
    return activation
    
def plot(activation, img_id, run_name, model_name, fold):
    vmin, vmax = get_min_max(activation)
    params = {'vmin': vmin, 'vmax': vmax, 'aspect': 'equal', 'cmap': 'RdYlGn'}
    fig, axn = plt.subplots(nrows=2, ncols=1, sharey=True)
    for ax, name in zip(axn.flat, ['reg_tanh', 'class_tanh']):
        # im = ax.imshow(np.random.random((10,10)), vmin=0, vmax=1)
        im = ax.imshow(activation[name], **params)
        
    for ax, name in zip(axn, ['Regression branch', 'Classification branch']):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.title.set_text(name)
    
    # # plt.imshow(activation['reg_tanh'], vmin=vmax
    # im1 = axn[0].imshow(activation['reg_tanh'], **params)
    # im2 = axn[1].imshow(activation['class_tanh'], **params)
    
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    cbar = fig.colorbar(im, ax=axn.ravel().tolist())
    cbar.set_ticks([])
    
    
    # fig.subplots_adjust(right=0.98)
    # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    # cbar_ax.tick_params(size=0)
    # fig.colorbar(im2, cax=cbar_ax)
    plt.savefig(osp.join(os.getcwd(), 'vis', 'fc_feature_map_run={}_model={}_fold={}_.png'.format(run_name, model_name, fold)), 
                format='png', dpi=300, bbox_inches='tight')
    
"""
Visualizes feature maps of first FC layers for given model. 
"""
def visualize_feature_maps(run_name, model_name, best_run_name, best_fold):
    (test_sample, args) = get_test_sample_args(run_name, best_run_name, best_fold)
    activation = generate_feature_maps(model_name, best_fold, test_sample['patch'], args)
    plot(activation, test_sample['img_id'], best_run_name, model_name, best_fold)

    
# # plot_params = {'xticklabels': False, 'yticklabels': False}
# print(activation['reg_tanh'].shape)

# plt.imshow(activation['reg_tanh'])
# plt.axis('off')
# plt.savefig(osp.join(os.getcwd(), 'vis', 'class.png'), format='png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    
    SAMPLE_IDS_FROM_RUN_NAME = '2021_07_01__11_23_50'
    best_run_name = '2021_07_07__23_02_22'
    model_name = 'best_test_score.pth'
    best_fold = 8
    
    visualize_feature_maps(run_name=SAMPLE_IDS_FROM_RUN_NAME, model_name=model_name, 
                           best_run_name=best_run_name, best_fold=best_fold)

    # patch_size, num_classes, split_layer, num_samples = 3, 12, 4, 1
    # in_channels = 12
    # use_atrous_conv, reshape_to_mosaic = False, False
    
    # model = EAOriginalDAN(in_channels=in_channels, patch_size=patch_size, 
    #                       split_layer=split_layer, num_classes=num_classes,
    #                       use_atrous_conv=use_atrous_conv,
    #                       reshape_to_mosaic=reshape_to_mosaic)
    
    # activation = {}
    # model.regressor[0][2].register_forward_hook(get_activation(activation=activation, name='tanh', task_name='reg'))
    # model.classifier[0][2].register_forward_hook(get_activation(activation=activation, name='tanh', task_name='class'))
    
    # inp = torch.randn(num_samples, in_channels, patch_size, patch_size)
    # outp = model(inp)
    
    # activation['reg_tanh'] = activation['reg_tanh'].view(8, -1)
    # activation['class_tanh'] = activation['class_tanh'].view(8, -1)

    ## plot(activation)
    ## sample = get_test_sample(SAMPLE_IDS_FROM_RUN_NAME, best_run_name, best_fold)
