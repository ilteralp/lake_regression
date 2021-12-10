#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:52:04 2021

@author: melike
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path as osp
import torch


"""
Takes preds and targets tensor and saves them into separate readable files. 
"""
def save_estimates_targets(preds, targets, run_name, model_name, fold):
    if preds.device.type == 'cuda':
        targets = targets.cpu()
        preds = preds.cpu()
    
    for t, n in zip([preds, targets], ['preds', 'targets']):
        path = osp.join(os.getcwd(), 'vis', 'run={}_model={}_fold={}_{}.txt'.format(run_name, model_name, fold, n))
        np.savetxt(path, t.numpy())

"""
Takes preds and targets tensor, plots them where targets form the baseline and
saves the plot. 
"""
def plot_estimates_targets(preds, targets, run_name, model_name, fold):
    if preds.device.type == 'cuda':
        targets = targets.cpu()
        preds = preds.cpu()
    
    fig, ax = plt.subplots()
    ax.scatter(targets, preds)
    ax.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--', lw=4)
    ax.set_xlabel('Observed')
    ax.set_ylabel('Estimated')
    path = osp.join(os.getcwd(), 'vis', 'run={}_model={}_fold={}_R2.png'.format(run_name, model_name, fold))
    plt.savefig(path)
    plt.show()
    
    
if __name__ == "__main__":
    num_values = 100
    # y = np.arange(0, num_values)
    # predicted = np.random.randint(0, num_values, num_values)
    # y = np.random.rand(num_values)
    # predicted = np.random.rand(num_values)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    y = torch.randn(num_values, device=device)
    predicted = torch.randn(num_values, device=device)
    
    # pred_path = osp.join(os.getcwd(), 'vis', 'preds.txt')
    # y_path = osp.join(os.getcwd(), 'vis', 'targets.txt')
    # np.savetxt(pred_path, predicted.numpy())
    
    # loaded_preds = np.loadtxt(pred_path, dtype=float)
    # print(predicted)
    # print(loaded_preds)
    
    # # np.savetxt('my_file.txt', torch.Tensor([3,4,5,6]).numpy())
    
    
    if device.type == 'cuda':
        y = y.cpu()
        predicted = predicted.cpu()
    
    fig, ax = plt.subplots()
    ax.scatter(y, predicted)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    PATH = osp.join(os.getcwd(), 'tmp.png')
    plt.savefig(PATH, format='png', dpi=300)
    plt.show()

