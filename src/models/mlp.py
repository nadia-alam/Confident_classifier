#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:31:39 2019

@author: nadia14
"""


import torch.nn as nn
class MLP(nn.Module):
    '''Class for Masked MLP with ReLu Activations'''
    def __init__(self, layer_dims):
        
        super(MLP, self).__init__()        
        self.dims = layer_dims
        self.linears = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i+1]) for i in range(len(self.dims)-1)])
        
        
#        linears = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
    def forward(self, x):
        relu = nn.ReLU(inplace=True)

        out = x.view(x.size(0), -1)
        #print(out.shape)
        for i in range(len(self.linears)-1):
            #print(i, out.shape)
            out = relu(self.linears[i](out))
        
        out = self.linears[-1](out)
        return out