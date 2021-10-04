import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import warnings
warnings.filterwarnings("ignore")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Dataset(data.Dataset):
    def __init__(self, input_length, mid, output_length, direc, support_idx, train_idx):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc

        self.support_ids = support_idx
        self.train_ids = train_idx

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, index):
        T_ID = self.train_ids[index]
        S_ID = np.random.choice(self.support_ids)
        
        t_y = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        t_x = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1, t_y.shape[-2], t_y.shape[-1])
        
        s_y = torch.load(self.direc + "/sample_" + str(S_ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        s_x = torch.load(self.direc + "/sample_" + str(S_ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1, s_y.shape[-2], s_y.shape[-1])
        
        
        return t_x.float(), t_y.float(), s_x.float(), s_y.float()
    
def train_epoch(train_loaders, model, optimizer, loss_function):
    train_mse = []
    k = 0
    for i, data_loader in enumerate(train_loaders):
        print(i)
        for t_xx, t_yy, s_xx, s_yy in data_loader:
            t_xx = t_xx.to(device)
            t_yy = t_yy.to(device)
            s_xx = s_xx.to(device)
            s_yy = s_yy.to(device)
            
            ims = model(s_xx, s_yy, t_xx)

            loss = loss_function(ims, t_yy.reshape(t_yy.shape[0], -1, t_yy.shape[3], t_yy.shape[4])) 
           
            train_mse.append(loss.item()) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loaders, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
   # with torch.no_grad():
    for i, data_loader in enumerate(valid_loaders):
        for t_xx, t_yy, s_xx, s_yy in data_loader:
            t_xx = t_xx.to(device)
            t_yy = t_yy.to(device)
            s_xx = s_xx.to(device)
            s_yy = s_yy.to(device)


            ims = model(s_xx, s_yy, t_xx)

            loss = loss_function(ims, t_yy.reshape(t_yy.shape[0], -1, t_yy.shape[3], t_yy.shape[4])) 

            preds.append(ims.cpu().data.numpy().reshape(ims.shape[0],-1,1,ims.shape[2],ims.shape[3]))
            trues.append(t_yy.cpu().data.numpy())
            valid_mse.append(loss.item())
    preds = np.concatenate(preds, axis = 0)  
    trues = np.concatenate(trues, axis = 0)  
    valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues


def test_epoch(test_loaders, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
   # with torch.no_grad():
    for i, data_loader in enumerate(test_loaders):
        for t_xx, t_yy, s_xx, s_yy in data_loader:
            t_xx = t_xx.to(device)
            t_yy = t_yy.to(device)
            s_xx = s_xx.to(device)
            s_yy = s_yy.to(device)


            ims = model(s_xx, s_yy, t_xx, test = True)

            loss = loss_function(ims, t_yy.reshape(t_yy.shape[0], -1, t_yy.shape[3], t_yy.shape[4])) 

            preds.append(ims.cpu().data.numpy().reshape(ims.shape[0],-1,1,ims.shape[2],ims.shape[3]))
            trues.append(t_yy.cpu().data.numpy())
            valid_mse.append(loss.item())
    preds = np.concatenate(preds, axis = 0)  
    trues = np.concatenate(trues, axis = 0)  
    valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues