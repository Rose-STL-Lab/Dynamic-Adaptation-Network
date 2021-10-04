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
    def __init__(self, input_length, mid, output_length, direc, lst_idx):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc

        self.support_ids = lst_idx
        self.train_ids = lst_idx

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, index):
        T_ID = self.train_ids[index]
        y = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        x = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1, y.shape[-2], y.shape[-1])
       
        return x.float(), y.float()
    
def train_epoch(train_loaders, model, optimizer, loss_function):
    train_mse = []
    k = 0
    for c, data_loader in train_loaders:
        for xx, yy in data_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            loss = 0
            for y in yy.transpose(0,1):
                im, c_pred = model(xx)
                xx = torch.cat([xx[:, im.shape[1]:], im], 1)
                loss += loss_function(im, y) + 0.5*loss_function(c_pred, torch.zeros(c_pred.shape).fill_(c).to(c_pred.device))
            train_mse.append(loss.item()/yy.shape[1]) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loaders, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for c, data_loader in valid_loaders:
            for xx, yy in data_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                loss = 0
                ims = []
                for y in yy.transpose(0,1):
                    im, c_pred = model(xx)
                    xx = torch.cat([xx[:, im.shape[1]:], im], 1)
                    loss += loss_function(im, y) 
                    ims.append(im.cpu().data.numpy())

                ims = np.array(ims).transpose(1,0,2,3,4)
                preds.append(ims)
                trues.append(yy.cpu().data.numpy())
                valid_mse.append(loss.item()/yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues