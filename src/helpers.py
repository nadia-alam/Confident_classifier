#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:49:19 2019

@author: nadia
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def plot_xrange_shades2(ax,sampling_regions, ds_1_parts, ds_2_parts):
    ds1_x_ranges =  sampling_regions[ds_1_parts]
    ds2_x_ranges =  sampling_regions[ds_2_parts]
    for m,i in enumerate(ds1_x_ranges):
        if m==0:
            ax.axvspan(i[0],i[1], alpha= 0.2, label = 'model 1 training region')
        else: 
            ax.axvspan(i[0],i[1], alpha= 0.2)
                
    for m,i in enumerate(ds2_x_ranges):
        if m==0:
            ax.axvspan(i[0],i[1], alpha= 0.2, color='pink', label = 'model 2 training region')
        else: 
            ax.axvspan(i[0],i[1], alpha= 0.2, color='pink')
            
            
def plot_xrange_shades(ax,sampling_regions, parts, c='green'):
    x_ranges =  sampling_regions[parts]
    
    for m,i in enumerate(x_ranges):
        if m==0:
            ax.axvspan(i[0],i[1], alpha= 0.2, color = c, label = 'in distribution regions')
        else: 
            ax.axvspan(i[0],i[1], alpha= 0.2, color = c)
                
    
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def seed_torch(seed=1029):
    '''This functions makes experiments repeatable in pytorch. Exclude last to lines to get better training efficiency , training performance may not be repeatable in that case'''

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def train(model, device, loss_fn, optimizer, max_iteration=1000, max_epoch=None, loader_train=None, loader_val=None, results_store_interval = 100, verbose=False, save_header=None):



    model.train()

    train_acc = torch.zeros(max_iteration//results_store_interval+1).to(device, non_blocking=True)
    val_acc = torch.zeros(max_iteration//results_store_interval+1).to(device, non_blocking=True)
    val_loss = torch.zeros(max_iteration//results_store_interval+1).to(device, non_blocking=True)
    iter_id = 0
    epoch = 0
    l = 0
    while iter_id<=max_iteration:
        #print('Starting epoch %d ' % (epoch + 1))
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        

        for t, (x, y) in enumerate(loader_train):

            #x_var, y_var = to_var(x), to_var(y.long())
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            scores = model(x)
            loss = loss_fn(scores, y)

            if (iter_id) % results_store_interval == 0:
                train_acc[l] = test(model, loader_train, None,  device, do_print=False)
                if loader_val:
                    val_acc[l], val_loss[l] = test(model,loader_val, loss_fn, device, do_print=False)

                    if verbose and (iter_id) % 200 == 0:
                        print('iteration = %d / %d, training loss = %.8f, validation loss = %.8f, training accuracy = %.3f, validation accuracy = %.3f' % (iter_id, max_iteration, loss.item(), val_loss[l], train_acc[l], val_acc[l]))

                else:
                    if verbose and (iter_id) % 200 == 0:
                        print('iteration = %d / %d, training loss = %.8f, training accuracy = %.3f ' % (iter_id, max_iteration, loss.item(), train_acc[l]))
                
                l+=1
                #print('l: ' , str(l))
            if (iter_id) % 5000 == 0 and iter_id>0 and save_header:
                print('saving model for iteration ' + str(iter_id))

                torch.save(model.state_dict(), save_header + '_' + str(iter_id) + '.pkl')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            iter_id+=1


            if iter_id>max_iteration:
                break
#
        epoch+=1
        if max_epoch:
            if epoch>max_epoch:
                
                break
    print('Trained for ' + str(epoch) + ' epochs')
    
    return train_acc, val_acc, val_loss
def get_k_fold_indices(train_size, folds=5):
    
    #folds = 5
    permuted_indices = torch.randperm(train_size).tolist()
    min_fold_size = train_size//folds
    remaining_points = train_size - min_fold_size * (folds) # if we uuse folds with min_folds_size, this number od folds should have 1 point more than min size
   
    last = -1
    test_idx = []
    train_idx = []
    for i in range(folds):
        if i < remaining_points:
            test_list = [m + last +1 for m in range(min_fold_size+1)]
        else:
            test_list = [m + last +1 for m in range(min_fold_size)]
        #print(test_list)
        last = test_list[-1]
        test_idx.append([permuted_indices[j] for j in test_list])
        #print(test_list)
        
        train_idx.append([l for l in permuted_indices if l not in test_idx[-1]])
        
        
        
    
    return train_idx,test_idx


def make_folder(dirName):

    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:    
        print("Directory " , dirName ,  " already exists")

def test(model, loader, loss_fun, device, return_score = False, do_print=True):

    num_correct, num_samples, loss_val, batch_cnt = 0, len(loader.dataset), 0, 0
    with torch.no_grad():

        for x, y in loader:
            x_var = x.to(device)
            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()
            if loss_fun:
                loss_val += loss_fun(scores, y.to(device)).item()
                batch_cnt +=1



        acc = float(num_correct) / num_samples
        if do_print:
            print('Test accuracy: {:.2f}% ({}/{})'.format(
            100.*acc,
            num_correct,
            num_samples,
            ))
    if loss_fun:
        if return_score:
            return acc, float(loss_val/batch_cnt), scores.data.cpu()
        return acc, float(loss_val/batch_cnt)
    if return_score:
        return acc, scores.data.cpu()
    return acc

def weights_init_linear_xavier_normal(m):
    '''initialize weights of linear FC filters
    '''
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        #nn.init.xavier_normal(m.bias.data)
def initialize_mask( model, layers_to_prune, dims, device):
    masks = []
    #initialize masks with 1
    for i,p in enumerate(layers_to_prune):
       masks.append(torch.ones([dims[i+1], dims[i]]).to(device))
    model.set_masks(masks)
    print('initial Masks set to 1')

# def prune_rate(model, verbose=True):
#     """
#     Print out prune rate for each layer and the whole network
#     """
#     total_nb_param = 0
#     nb_zero_param = 0

#     layer_id = 0

#     for parameter in model.parameters():

#         param_this_layer = 1
#         for dim in parameter.data.size():
#             param_this_layer *= dim
        

#         # only pruning linear and conv layers
#         if len(parameter.data.size()) != 1:
#             total_nb_param += param_this_layer
#             print('param_this_layer =' , param_this_layer)
#             layer_id += 1
#             zero_param_this_layer = \
#                 np.count_nonzero(parameter.cpu().data.numpy()==0)
#             nb_zero_param += zero_param_this_layer
#             print('zero_param_this_layer =' , zero_param_this_layer)
#             if verbose:
#                 print("Layer {} | {} layer | {:.2f}% parameters pruned" \
#                     .format(
#                         layer_id,
#                         'Conv' if len(parameter.data.size()) == 4 \
#                             else 'Linear',
#                         100.*zero_param_this_layer/param_this_layer,
#                         ))
#     pruning_perc = 100.*nb_zero_param/total_nb_param
#     if verbose:
#         print("Final pruning rate: {:.2f}%".format(pruning_perc)+ str(total_nb_param))
#     return pruning_perc



def prune_rate(model, layers_to_prune, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for l in layers_to_prune:

        param_this_layer = 1
        for dim in l.mask.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer
        print('param_this_layer =' , param_this_layer)
        layer_id += 1
        zero_param_this_layer = torch.sum(l.mask.data == 0).item()
        nb_zero_param += zero_param_this_layer
        print('zero_param_this_layer =' , zero_param_this_layer)
        if verbose:
            print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(l.mask.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc)+ str(total_nb_param))
    return pruning_perc



def min_VL_iters(dataset_name):
    return list(np.load('logs/argmin_VLs_' + dataset_name + '_mlp-2.npy'))
#
#def write_results(header,train_accuracies,valid_accuracies, new_csv=False):
#    from os.path import isfile
#    path = 'results/'
#    csv_file = path + "results.csv"
#    if new_csv or not isfile(csv_file):
#        f = open(csv_file, 'w')
#        f.close()
#        row = ['prunning_percent', 'early_stopping', 'train_acc_mean', 'train_acc_stdev','train_acc_mean', 'train_acc_stdev']
#        with open(csv_file, 'a') as csvFile:
#            writer = csv.writer(csvFile)
#            writer.writerow(row)
#        csvFile.close()
#
#
#
#    row =  ['dataset', 'training_accuracy', 'test_accuracy']
#    with open(csv_file, 'a') as csvFile:
#        writer = csv.writer(csvFile)
#        writer.writerow(row)
#        csvFile.close()
#
#
#    for file in dirs:
#        file_path = os.path.join(data_path,file)
#        dataset_name = os.path.splitext(file)[0]
#        path = 'saved/' +dataset_name
#        try:
#            os.mkdir(path)
#        except:
#            pass
#        if write_csv!='skip':
#            if write_csv=='temp':
#                csv_file = path + '/' + dataset_name + "_temp.csv"
#            else:
#                csv_file = path + '/' + dataset_name + ".csv"
#
#            if write_csv== 'overwrite':
#                f = open(csv_file, 'w')
#                f.close()
#                row = ['partition', 'seed', 'training_accuracy', 'test_accuracy', 'best_test_accuracy_of_subnetworks', 'ratio_accuracy_bestsub_to_original', 'average_error_correlation_of_subnetwork']
#                with open(csv_file, 'a') as csvFile:
#                    writer = csv.writer(csvFile)
#                    writer.writerow(row)
#                csvFile.close()
