import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import sampler

from os.path import join
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#def to_var(x, requires_grad=False, volatile=False):
#    """
#    Varialbe type that automatically choose cpu or cuda
#    """
#    if torch.cuda.is_available():
#        x = x.cuda()
#    return Variable(x, requires_grad=requires_grad, volatile=volatile)

    
#def train(model, loss_fn, optimizer, param, loader_train, loader_val=None):
#
#    model.train()
#    for epoch in range(param['num_epochs']):
#        print('Starting epoch %d / %d' % (epoch + 1, param['num_epochs']))
#
#        for t, (x, y) in enumerate(loader_train):
#            #x_var, y_var = to_var(x), to_var(y.long())
#            x, y = x.to(device), y.to(device)
#            scores = model(x)
#            loss = loss_fn(scores, y)
#
#            if (t + 1) % 100 == 0:
#                print('t = %d, loss = %.8f' % (t + 1, loss.item()))
#
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()

            
            
def train(model, loss_fn, optimizer, max_iteration=1000, max_epoch=None, loader_train=None, loader_val=None):

    
    
    model.train()

    train_acc = []
    val_acc = []

    iter_id = 0
    epoch = 0
    while iter_id<max_iteration:
        #print('Starting epoch %d ' % (epoch + 1))

        for t, (x, y) in enumerate(loader_train):
            
            #x_var, y_var = to_var(x), to_var(y.long())
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)

            if (iter_id + 1) % 100 == 0:
                train_acc.append(test(model, loader_train, do_print=False))
                if loader_val:
                    val_acc.append(test(model,loader_val, do_print=False))
                    #print('iteration = %d, loss = %.8f, training accuracy = %.3f, test accuracy = %.3f' % (iter_id + 1, loss.item(), train_acc[-1], val_acc[-1]))

                #else:
                    #print('iteration = %d, loss = %.8f, training accuracy = %.3f' % (iter_id + 1, loss.item(), train_acc[-1]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            iter_id+=1
            
            if iter_id>=max_iteration:
                break
#        
        epoch+=1
        if max_epoch:
            if epoch>max_epoch:
                break
    #print(iter_id)
    return train_acc, val_acc

def test(model, loader, do_print=True):

    model.eval()

    num_correct, num_samples = 0, len(loader.dataset)
    with torch.no_grad():
            
        for x, y in loader: 
            x_var = x.to(device)
            scores = model(x_var)
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == y).sum()

        acc = float(num_correct) / num_samples
        if do_print:
            print('Test accuracy: {:.2f}% ({}/{})'.format(
            100.*acc,
            num_correct,
            num_samples,
            ))
    
    return acc
    

def prune_rate(model, verbose=True):
    """
    Print out prune rate for each layer and the whole network
    """
    total_nb_param = 0
    nb_zero_param = 0

    layer_id = 0

    for parameter in model.parameters():

        param_this_layer = 1
        for dim in parameter.data.size():
            param_this_layer *= dim
        total_nb_param += param_this_layer

        # only pruning linear and conv layers
        if len(parameter.data.size()) != 1:
            layer_id += 1
            zero_param_this_layer = \
                np.count_nonzero(parameter.cpu().data.numpy()==0)
            nb_zero_param += zero_param_this_layer

            if verbose:
                print("Layer {} | {} layer | {:.2f}% parameters pruned" \
                    .format(
                        layer_id,
                        'Conv' if len(parameter.data.size()) == 4 \
                            else 'Linear',
                        100.*zero_param_this_layer/param_this_layer,
                        ))
    pruning_perc = 100.*nb_zero_param/total_nb_param
    if verbose:
        print("Final pruning rate: {:.2f}%".format(pruning_perc))
    return pruning_perc


def arg_nonzero_min(a):
    """
    nonzero argmin of a non-negative array
    """

    if not a:
        return

    min_ix, min_v = None, None
    # find the starting value (should be nonzero)
    for i, e in enumerate(a):
        if e != 0:
            min_ix = i
            min_v = e
    if not min_ix:
        print('Warning: all zero')
        return np.inf, np.inf

    # search for the smallest nonzero
    for i, e in enumerate(a):
         if e < min_v and e != 0:
            min_v = e
            min_ix = i

    return min_v, min_ix
    
