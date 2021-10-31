#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 16:01:22 2021

@author: melike

From https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
"""

import torch
import torch.nn as nn
from torch.distributions.gumbel import Gumbel
import math
import numpy as np

ONE_DIV_SQRT_TWO_PI = 1.0 / np.sqrt(2.0*np.pi) # normalization factor for Gaussians

class MaruMDN(nn.Module):
    def __init__(self, in_channels, patch_size, n_hidden, n_gaussians):
        super(MaruMDN, self).__init__()
        self.in_features = in_channels * patch_size * patch_size
        
        self.z_h = nn.Sequential(
            nn.Linear(self.in_features, n_hidden),
            nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)  

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1)
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu
    
    @staticmethod
    def gaussian_distribution(y, mu, sigma):
        # make |mu|=K copies of y, subtract mu, divide by sigma
        result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
        result = -0.5 * (result * result)
        return (torch.exp(result) * torch.reciprocal(sigma)) * ONE_DIV_SQRT_TWO_PI
    
    @staticmethod
    def mdn_loss(pi, sigma, mu, y):
        result = MaruMDN.gaussian_distribution(y, mu, sigma) * pi
        result = torch.sum(result, dim=1)
        result = -torch.log(result)
        return torch.mean(result)

    @staticmethod
    def gumbel_sample(x, axis=1):
        z = torch.from_numpy(np.random.gumbel(loc=0, scale=1, size=x.shape)).to(x.device)
        return torch.argmax(torch.log(x) + z, dim=axis)
        # z = Gumbel(loc=torch.tensor([0.0]), scale=torch.tensor([1.0])).expand(x.shape).sample().to(x.device)
        # return torch.argmax((torch.log(x) + z), dim=axis)
    
    @staticmethod
    def get_pred(pi_data, sigma_data, mu_data, n_samples):
        k = MaruMDN.gumbel_sample(pi_data)
        indices = (torch.from_numpy(np.arange(n_samples)).to(pi_data.device), k)
        rn = torch.from_numpy(np.random.randn(n_samples)).to(pi_data.device)
        # indices = (torch.arange(n_samples), k)
        # rn = torch.randn(n_samples, device=pi_data.device)
        sampled = rn * sigma_data[indices] + mu_data[indices]
        return sampled
        
        