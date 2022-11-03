# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 14:48:24 2022

@author: engrh
"""

import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return 1-np.tanh(x)**2;

def relu(x):
    return np.maximum(x,0);

def sigmoid(x):
    return 1/(1+np.exp(-x));