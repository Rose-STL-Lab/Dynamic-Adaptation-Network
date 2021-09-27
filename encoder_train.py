from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import time
import os
from torch.autograd import Variable
from utils import train_epoch, eval_epoch, Dataset, get_lr
from models import Encode_Style_3D
import random
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


vor_factors = np.load("PhiFlow/task_parameter_vorticity_turbulence.npy")

name = "encoder"
input_length = 20
batch_size = 64
num_epoch = 1000
learning_rate = 0.001
min_mse = 1
out_length = 3
# Coefficients of loss terms
alpha, beta = 1. , 1.

test_factors = [3, 8, 13, 18, 23]
factors = list(set(list(range(1,26))) - set(test_factors))


direc = "PhiFlow/sliced_data/data"
train_loaders = [(vor_factors[i], data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, direc = direc + str(factor), 
                                                          lst_idx = list(range(0,350)), stack = False),  batch_size = batch_size, shuffle = True, num_workers = 8)) 
                 for i, factor in enumerate(factors)]

valid_loaders = [(vor_factors[factor-1], data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, 
                                                                 direc = direc + str(factor), lst_idx = list(range(350, 400)), stack = False),  
                                 batch_size = batch_size, shuffle = False, num_workers = 8)) for factor in factors]


encoder = nn.DataParallel(Encode_Style_3D(in_channels = 2, style_dim = 512).to(device))
optimizer = torch.optim.Adam(encoder.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()

train_mse = []
valid_mse = []

for i in range(num_epoch):
    start = time.time()
    scheduler.step()
    train_losses, valid_losses = [],[]
    encoder.train()
    for fac, data_loader in train_loaders:
        loss = 0
        ims = []
        for xx, yy in data_loader:
            xx = xx.to(device)
            c, z = encoder(xx)
            ims = z
            pairwise_loss = loss_fun(ims[1:], ims[:-1])
            mag_loss = loss_fun(torch.mean(ims**2, dim = 1), torch.ones(ims.shape[0]).to(ims.device))
            fac_loss = loss_fun(c, torch.zeros(c.shape).fill_(fac).to(device))
            loss = pairwise_loss + alpha * mag_loss + beta * fac_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
    encoder.eval()
    with torch.no_grad():
        for fac, data_loader in valid_loaders:
            loss = 0
            ims = []
            for xx, yy in data_loader:
                xx = xx.to(device)
                c, z = encoder(xx)
                ims = z
                pairwise_loss = loss_fun(ims[1:], ims[:-1])
                mag_loss = loss_fun(torch.mean(ims**2, dim = 1), torch.ones(ims.shape[0]).to(ims.device))
                fac_loss = loss_fun(c, torch.zeros(c.shape).fill_(fac).to(device))
                loss = pairwise_loss + alpha * mag_loss + beta * fac_loss
                valid_losses.append(loss.item())
    train_mse.append(np.mean(train_losses))
    valid_mse.append(np.mean(valid_losses))
    
    print(i, train_mse[-1], valid_mse[-1])
    if np.mean(valid_losses) < min_mse:
        min_mse = np.mean(valid_losses)
        best_model = encoder
        torch.save(best_model, name + ".pth")
        
    if (len(train_mse) > 100 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break       
        