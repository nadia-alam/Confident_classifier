##############################################
# This code is based on samples from pytorch #
##############################################
# Writer: Based on code from Kimin Lee 
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import data_loader
import numpy as np
#import torchvision.utils as vutils
import models
import matplotlib.pyplot as plt
#from torch.utils.serialization import load_lua
#from torchvision import datasets, transforms
from torch.autograd import Variable
from toy_2d import toy_inD, toy_outD
from torch.utils import data as D
# Create toy data
# -*- coding: utf-8 -*-

#import numpy as np
# configs
model_dims = [2, 500, 500, 2]


# data
x_in, y_in = toy_inD(10000)
x_out = toy_outD(10000,x_min=[-10, -10], x_max=[10, 10])
#plt.scatter(x_in[:,0], x_in[:,1], s=2, c=y_in
#plt.scatter(x_out[:,0], x_out[:,1], s=2, marker='x')


# Training settings
parser = argparse.ArgumentParser(description='Training code - joint confidence')
parser.add_argument('--batch-size', type=int, default=400, help='input batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.002, help='learning rate')
#parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--log-interval', type=int, default=1, help='how many batches to wait before logging training status')
#parser.add_argument('--dataset', default='svhn', help='cifar10 | svhn')
#parser.add_argument('--dataroot', required=True, help='path to dataset')
#parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
parser.add_argument('--outf', default='save_dir_beta_2', help='folder to output images and model checkpoints')
#parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--droprate', type=float, default=0.1, help='learning rate decay')
parser.add_argument('--decreasing_lr', default='5', help='decreasing strategy')
parser.add_argument('--num_classes', type=int, default=2, help='the # of classes')
parser.add_argument('--beta', type=float, default=1, help='penalty parameter for KL term')

args = parser.parse_args()

print(args)

print("Random Seed: ", args.seed)

if torch.cuda.is_available() :
    device = torch.device('cuda')
    torch.cuda.manual_seed(args.seed)
    kwargs = {'num_workers': 1, 'pin_memory': True} 
else:
    device = torch.device('cpu')
    torch.manual_seed(args.seed)
    kwargs = {} 

fig = plt.figure(figsize=(14,14))

print('load data: ')
train_loader = D.DataLoader(D.TensorDataset(torch.Tensor(x_in), torch.Tensor(y_in.astype(int)), torch.Tensor(x_out)),shuffle=True, batch_size=args.batch_size)
print('Load model')
model = models.MLP(model_dims)
print(model)

def test_plot():
    x = torch.tensor(toy_outD(100000)).to(device).float()
#    x = torch.meshgrid([torch.arange(-10,10,.1), torch.arange(-10,10,.1)])
#    x = torch.cat((x[0].reshape(-1,1),x[1].reshape(-1,1)),1).to(device)
    y = torch.exp(logSm(model(x)))
    x = x.cpu().data.numpy()
    y = y.cpu().data.numpy()
    x_class_0 = x[y[:,0]<0.2]
    x_class_1 = x[y[:,0]>0.2]
    #print(y.shape)
   
    plt.clf()
    #plt.figure(figsize=[15,15],)
    plt.scatter(x[:,0], x[:,1], s=2, c=y[:,0], cmap='gist_earth')
    plt.colorbar()
    col = ['red', 'darkblue']
    plt.scatter(x_in[:,0], x_in[:,1], s=2, c=[col[i] for i in y_in.astype(int)])
    #fig.show()
    fig.canvas.draw()
model.to(device)
#criterion.to(device)

print('Setup optimizer')
optimizer = optim.Adam(model.parameters(), lr=args.lr)

decreasing_lr = list(map(int, args.decreasing_lr.split(',')))


def test_plot2():
    x = torch.tensor(toy_outD(100000)).to(device).float()
#    x = torch.meshgrid([torch.arange(-10,10,.1), torch.arange(-10,10,.1)])
#    x = torch.cat((x[0].reshape(-1,1),x[1].reshape(-1,1)),1).to(device)
    y = torch.exp(logSm(model(x)))
    x = x.cpu().data.numpy()
    y = y.cpu().data.numpy()
    x_class_0 = x[y[:,0]<0.2]
    x_class_1 = x[y[:,0]>0.8]
    x_class_no = x[(y[:,0]>0.2) & (y[:,0]<0.8)]
#    #print(y.shape)
    
    plt.clf()
    #plt.figure(figsize=[15,15],)

#    col = ['red', 'darkblue']
    plt.scatter(x_in[y_in==0,0], x_in[y_in==0,1], s=20, marker='*',color='white', edgecolors= "black", label='class 0 training points')
    plt.scatter(x_in[y_in==1,0], x_in[y_in==1,1], s=20, marker='<',color='white', edgecolors= "black", label='class 1 training points')
    plt.scatter(x_class_0[:,0], x_class_0[:,1], s=1, c='firebrick', label="class 0 prob <0.2", alpha=0.7)
    plt.scatter(x_class_1[:,0], x_class_1[:,1], s=1, c='royalblue', label="class 0 prob >0.8", alpha=0.7)
    plt.scatter(x_class_no[:,0], x_class_no[:,1], s=1, c='yellowgreen', label="class 0 prob in [0.2,0.8]", alpha=0.7)
    plt.legend()
    
    #fig.show()
    
    fig.canvas.draw()

#data, target, data_out = next(iter(train_loader))
logSm = nn.LogSoftmax(dim=1)
def train(epoch):
    model.train()
    #for batch_idx, (data, target, data_out) in enumerate(train_loader):
    x_in, y_in = toy_inD(400)
    x_out = toy_outD(400,x_min=[-10, -10], x_max=[10, 10])
    data, target, data_out = torch.Tensor(x_in), torch.Tensor(y_in.astype(int)), torch.Tensor(x_out)
    uniform_dist = torch.Tensor(data.size(0), args.num_classes).fill_((1./args.num_classes)).to(device)
    data, target, data_out = data.to(device), target.long().to(device), data_out.to(device)
      
        

    ###########################
    #  Update classifier   #
    ###########################
    # cross entropy loss
    optimizer.zero_grad()
    output = logSm(model(data))
    loss = F.nll_loss(output, target)

    # KL divergence

    ood_model_output = logSm(model(data_out))
    KL_loss = F.kl_div(ood_model_output, uniform_dist, reduction='batchmean')*args.num_classes
    total_loss = loss + args.beta * KL_loss
    total_loss.backward()
    optimizer.step()

    #if batch_idx % 10 == 0:
    print('Classification Train Epoch: {},  Loss: {:.6f}, KL fake Loss: {:.6f}'.format(
            epoch, loss.item(), KL_loss.item()))
        #print(torch.exp(ood_model_output ))
#            vutils.save_image(fake.data, '%s/gan_samples_epoch_%03d.png'%(args.outf, epoch), normalize=True)\

    test_plot2()

#   
#def test(epoch):
#    model.eval()
#    test_loss = 0
#    correct = 0
#    total = 0
#    for data, target in test_loader:
#        total += data.size(0)
#        if args.cuda:
#            data, target = data.to(device), target.to(device)
#        data, target = Variable(data, volatile=True), Variable(target)
#        output = F.log_softmax(model(data))
#        test_loss += F.nll_loss(output, target).data[0]
#        pred = output.data.max(1)[1] # get the index of the max log-probability
#        correct += pred.eq(target.data).cpu().sum()
#
#    test_loss = test_loss
#    test_loss /= len(test_loader) # loss function already averages over batch size
#    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#        test_loss, correct, total,
#        100. * correct / total))


for epoch in range(1, args.epochs + 1):
    train(epoch)
#    test(epoch)
    if epoch in decreasing_lr:
#        optimizerG.param_groups[0]['lr'] *= args.droprate
#        optimizerD.param_groups[0]['lr'] *= args.droprate
        optimizer.param_groups[0]['lr'] *= args.droprate
    if epoch % 20 == 0:
        # do checkpointing
#        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outf, epoch))
#        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outf, epoch))
        torch.save(model.state_dict(), '%s/model_epoch_%d.pth' % (args.outf, epoch))


