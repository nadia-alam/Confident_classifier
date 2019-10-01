# -*- coding: utf-8 -*-


#import torch
import torch.nn as nn
from pruning.layers import MaskedLinear #, MaskedConv2d 



#
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = MaskedLinear(28*28, 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = MaskedLinear(200, 200)
        self.relu2 = nn.ReLU(inplace=True)
        self.linear3 = MaskedLinear(200, 10)
        
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.relu1(self.linear1(out))
        out = self.relu2(self.linear2(out))
        out = self.linear3(out)
        return out

    def set_masks(self, masks):
        # Should be a less manual way to set masks
        # Leave it for the future
        self.linear1.set_mask(masks[0])
        self.linear2.set_mask(masks[1])
        self.linear3.set_mask(masks[2])
    

import torch.nn as nn



class MaskedMLP(nn.Module):
    '''Class for Masked MLP with ReLu Activations'''
    def __init__(self, layer_dims):
        
        super(MaskedMLP, self).__init__()        
        self.dims = layer_dims
        self.linears = nn.ModuleList([MaskedLinear(self.dims[i], self.dims[i+1]) for i in range(len(self.dims)-1)])
        
        
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
    
    def set_masks(self, masks):

        
        for i in range(len(self.linears)):
            self.linears[i].set_mask(masks[i])
        



class Generator(nn.Module):
    def __init__(self, layer_dims):
        super(Generator, self).__init__()
        self.dims = layer_dims     
        self.linears = nn.ModuleList([MaskedLinear(self.dims[i], self.dims[i+1]) for i in range(len(self.dims)-1)])  

    # forward method
    def forward(self, x): 
        #l_relu = nn.Leaky
        out = x.view(x.size(0), -1)
        for i in range(len(self.linears)-1):
            out = nn.LeakyReLU(self.linears[i](out), 0.2)
        
        return torch.tanh(self.linears[-1](out))


    # for later using with pruning

    def set_masks(self, masks):

        
        for i in range(len(self.linears)):
            self.linears[i].set_mask(masks[i])
        

    
class Discriminator(nn.Module):
    def __init__(self, layer_dims):
        super(Discriminator, self).__init__()
        self.dims = layer_dims     
        self.linears = nn.ModuleList([MaskedLinear(self.dims[i], self.dims[i+1]) for i in range(len(self.dims)-1)])  
    
    # forward method
    def forward(self, x):
        out = x.view(x.size(0), -1)
        for i in range(len(self.linears)-1):
            out = nn.LeakyReLU(self.linears[i](out), 0.2)
            out = nn.Dropout(out, 0.3)
        
        return torch.sigmoid(self.linears[-1](out))

    # for later using with pruning
        
    def set_masks(self, masks):

        
        for i in range(len(self.linears)):
            self.linears[i].set_mask(masks[i])
        


# class Generator(nn.Module):
#     def __init__(self, g_input_dim, g_output_dim):
#         super(Generator, self).__init__()       
#         self.fc1 = nn.Linear(g_input_dim, 256)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features*2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features*2)
#         self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)
    
#     # forward method
#     def forward(self, x): 
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         return torch.tanh(self.fc4(x))
    
# class Discriminator(nn.Module):
#     def __init__(self, d_input_dim):
#         super(Discriminator, self).__init__()
#         self.fc1 = nn.Linear(d_input_dim, 1024)
#         self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
#         self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
#         self.fc4 = nn.Linear(self.fc3.out_features, 1)
    
#     # forward method
#     def forward(self, x):
#         x = F.leaky_relu(self.fc1(x), 0.2)
#         x = F.dropout(x, 0.3)
#         x = F.leaky_relu(self.fc2(x), 0.2)
#         x = F.dropout(x, 0.3)
#         x = F.leaky_relu(self.fc3(x), 0.2)
#         x = F.dropout(x, 0.3)
#         return torch.sigmoid(self.fc4(x))


