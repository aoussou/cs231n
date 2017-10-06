import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import copy
import math
import os
file_dir= os.path.dirname(os.path.realpath(__file__))
os.chdir(file_dir)

class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    Arguments:
        num_samples: # of desired datapoints
        start: offset where we should start selecting from
    """
    def __init__(self, num_samples, start = 0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

NUM_TRAIN = 49000
NUM_VAL = 1000

cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                           transform=T.ToTensor())
loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True,
                          transform=T.ToTensor())
loader_test = DataLoader(cifar10_test, batch_size=64)


dtype = torch.FloatTensor # the CPU datatype

# Constant to control how frequently we print train loss
print_every = 100

# This is a little utility that we'll use to reset the model
# if we want to re-initialize all our parameters
def reset(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()
        
class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
     
#####################################################
# Verify that CUDA is properly configured and you have a GPU available
torch.cuda.is_available()
gpu_dtype = torch.cuda.FloatTensor

def train(model, loss_fn, optimizer, num_epochs = 1):
    for epoch in range(num_epochs):
        print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
        model.train()
        for t, (x, y) in enumerate(loader_train):
            x_var = Variable(x.type(gpu_dtype))
            y_var = Variable(y.type(gpu_dtype).long())

            scores = model(x_var)
            
            loss = loss_fn(scores, y_var)
            if (t + 1) % print_every == 0:
                print('t = %d, loss = %.4f' % (t + 1, loss.data[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def check_accuracy(model, loader):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
    for x, y in loader:
        x_var = Variable(x.type(gpu_dtype), volatile=True)

        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
    acc = float(num_correct) / num_samples
    print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

#####################################################
######## MY MODEL ##################################    
hw0 = 32 # height = width of the input 
nC = 3  # number of channels of the input

# assuming we're using the same parameters for each convolutional layer
ck = 3 # window size - "k" like "kernel"
p = 1 # padding 
f = 32 # number of filters
cs = 1 # stride (default PyTorch = 1)

# pooling
pk = 2

# heights = widths after the first convolutional layer 
hwc1 = math.floor((hw0 - ck + 2*p)/cs+1)

nConv = 3 # number of [conv-relu-batchnorm-MaxPool] layer groups

# heigh = width of the data "at the end"
# before the linear layers
hwe = int(hwc1/pk**nConv)


my_model_base = nn.Sequential(
                    nn.Conv2d(nC, f, ck,padding=p),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(f),
                    nn.MaxPool2d(pk)
            )

for i in range(nConv-1):
    my_model_base = nn.Sequential(my_model_base,
                        nn.Conv2d(f, f, ck,padding=p),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm2d(f),
                        nn.MaxPool2d(pk),
                )

my_model_base = nn.Sequential(my_model_base,
                    Flatten(),    
                    nn.Linear(hwe*hwe*f, 1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(p=0.4),
                    nn.Linear(1024,10)    
            )

#######################################################

torch.cuda.synchronize() # Make sure there are no pending GPU computations
model = copy.deepcopy(my_model_base).type(gpu_dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.RMSprop(model.parameters(), lr=1e-3) 

train(model, loss_fn, optimizer, num_epochs=10)
check_accuracy(model, loader_val)

check_accuracy(model, loader_test)