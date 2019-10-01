import numpy as np
def toy_inD(n=1000): 
    # in_dist data
    verts = [(1.5,1), (-1,-1)]
    
    pos = np.random.normal(size=(n, 2), scale = 0.2)
    labels = np.vstack((np.zeros([n//2,1]),np.ones([n//2,1])))
    for i,v in enumerate(verts):
        pos[i*n//2:(i+1)*n//2,:] += v
    shuffling = np.random.permutation(n)
    pos = pos[shuffling]
    labels = labels[shuffling]
        
    return pos, labels.reshape(-1)

def toy_outD(n=1000, x_min=[-20, -20], x_max=[20, 20]):
    
    pos = np.random.uniform(low=x_min, high=x_max, size=(n,2))
    
    return pos# -*- coding: utf-8 -*-

