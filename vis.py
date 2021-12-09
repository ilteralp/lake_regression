#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 16:52:04 2021

@author: melike
"""
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

num_values = 100
y = np.arange(0, num_values)
predicted = np.random.randint(0, num_values, num_values)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
