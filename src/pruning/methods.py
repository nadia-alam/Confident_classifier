import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pruning.utils import prune_rate, arg_nonzero_min


def weight_prune(model, pruning_perc):
    '''
    Prune pruning_perc% weights globally (not layer-wise)
    arXiv: 1606.09274
    
    '''  
    
    all_weights = []
    for p in model.parameters():
        print(p.data.size())
        if len(p.data.size()) != 1:
            all_weights += list(p.cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), pruning_perc)

    # generate mask
    masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            pruned_inds = p.data.abs() > threshold
            masks.append(pruned_inds.float())
    return masks


def prune_one_filter(model, masks):
    '''
    Pruning one least ``important'' feature map by the scaled l2norm of 
    kernel weights
    arXiv:1611.06440
    '''
    NO_MASKS = False
    # construct masks if there is not yet
    if not masks:
        masks = []
        NO_MASKS = True

    values = []
    for p in model.parameters():

        if len(p.data.size()) == 4: # nasty way of selecting conv layer
            p_np = p.data.cpu().numpy()

            # construct masks if there is not
            if NO_MASKS:
                masks.append(np.ones(p_np.shape).astype('float32'))

            # find the scaled l2 norm for each filter this layer
            value_this_layer = np.square(p_np).sum(axis=1).sum(axis=1)\
                .sum(axis=1)/(p_np.shape[1]*p_np.shape[2]*p_np.shape[3])
            # normalization (important)
            value_this_layer = value_this_layer / \
                np.sqrt(np.square(value_this_layer).sum())
            min_value, min_ind = arg_nonzero_min(list(value_this_layer))
            values.append([min_value, min_ind])

    assert len(masks) == len(values), "something wrong here"

    values = np.array(values)

    # set mask corresponding to the filter to prune
    to_prune_layer_ind = np.argmin(values[:, 0])
    to_prune_filter_ind = int(values[to_prune_layer_ind, 1])
    masks[to_prune_layer_ind][to_prune_filter_ind] = 0.

    print('Prune filter #{} in layer #{}'.format(
        to_prune_filter_ind, 
        to_prune_layer_ind))

    return masks


def filter_prune(model, pruning_perc):
    '''
    Prune filters one by one until reach pruning_perc
    (not iterative pruning)
    '''
    masks = []
    current_pruning_perc = 0.

    while current_pruning_perc < pruning_perc:
        masks = prune_one_filter(model, masks)
        model.set_masks(masks)
        current_pruning_perc = prune_rate(model, verbose=False)
        print('{:.2f} pruned'.format(current_pruning_perc))

    return masks



# Copyright (C) 2018 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A collection of pruning heuristics."""


def prune_by_percent(layers, percents, return_done_flag=False):
    """Return new masks that involve pruning the smallest of the final weights.

    Args:
        model: to prune
        percents: A list determining the percent by which to prune each layer.
    
        masks: A list of tensors containing the current masks.
    
    Returns:
        A list of newly-pruned masks.
    """

    def prune_by_percent_once(percent, mask, final_weight):
      
      
        # Put the weights that aren't masked out in sorted order.
    
        sorted_weights = torch.sort(torch.abs(final_weight[mask == 1]))[0]
        if sorted_weights.shape[0] <= 1:
            return mask, True
        
        # Determine the cutoff for weights to be pruned.
        cutoff_index = torch.round(torch.tensor(percent * sorted_weights.numel())).long()
        cutoff = sorted_weights[cutoff_index]
        
    
        # Prune all weights below the cutoff.
        return (final_weight.data.abs()>cutoff).float() * mask, False
        
    with torch.no_grad():
        new_masks = []
        all_layer_done = True
        for i, l in enumerate(layers):
            mask, done = prune_by_percent_once(percents[i], l.mask.data, l.weight.data)
            new_masks.append(mask)
            if not done:
                all_layer_done = False
                
            
              
              
    
    if return_done_flag:
        return new_masks, all_layer_done
    return new_masks
    
