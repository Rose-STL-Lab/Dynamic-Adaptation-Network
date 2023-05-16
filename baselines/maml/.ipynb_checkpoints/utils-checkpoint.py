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
import copy
from collections import OrderedDict
import random
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
    for data_loader in train_loaders:
        for xx, yy in data_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            #ss = ss.to(device)
            loss = 0
            for y in yy.transpose(0,1):
                im = model(xx)##, style, ss
                xx = torch.cat([xx[:, im.shape[1]:], im], 1)
                loss += loss_function(im, y) 
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
        for data_loader in valid_loaders:
            for xx, yy in data_loader:
                xx = xx.to(device)
                yy = yy.to(device)
                #ss = ss.to(device)
                loss = 0
                ims = []
                for y in yy.transpose(0,1):
                    im = model(xx)#, xx[:,:40],_, ss
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


class MAML_train():
    def __init__(self, data_tasks, model, meta_optimiser, scheduler, num_per_batch, inner_lr, meta_lr):
        self.model = model
        self.data_tasks = data_tasks
        self.loss_fun = nn.MSELoss()
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.weights = list(self.model.parameters())#self.model.state_dict()
        self.meta_optimiser = meta_optimiser
        self.num_per_batch = num_per_batch
        self.scheduler = scheduler
        
    def zero_grad(self, params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()
      

    def inner_loop(self, xx, yy):
        loss = 0
        for y in yy.transpose(0,1):
            im = self.model(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += self.loss_fun(im, y)/yy.shape[1]
            
        self.zero_grad(self.model.parameters())
        grads = torch.autograd.grad(loss, self.model.parameters())#, create_graph=True)
        mod_state_dict = self.model.clone_state_dict()
        mod_weights=OrderedDict()
        
        for (k,v),g in zip(self.model.named_parameters(), grads):
            mod_weights[k] = v - self.inner_lr*g
            mod_state_dict[k] = mod_weights[k]
            
        return mod_state_dict
    
    def duplicate(self, num, tensor):
#         lst = [tensor.clone().unsqueeze(0) for i in range(num)]
#         return torch.cat(lst, dim = 0)
        lst = [num] + [1] * len(tensor.shape)
        return tensor.repeat(*lst)
        
    def main_loop(self, num_epochs,  train_loaders, valid_loaders, name):
                
        train_mse = []
        valid_mse = []
        test_mse = []
        min_mse = 10
        
        for num in range(num_epochs):
            self.model.train()
            start = time.time()
            
            state_dicts=[]
            loaders_list=[]
            for batch in np.random.choice(len(self.data_tasks[0]), self.num_per_batch):#range(len(self.data_tasks[0])):
                tasks = list(range(len(self.data_tasks)))
                for t in tasks:
                    xx, yy = self.data_tasks[t][batch]
                    xx, yy = xx.to(device), yy.to(device)
                    
                    d = self.inner_loop(xx, yy)
                    state_dicts.append(d)
                    
                meta_loss = 0
                for i, t in enumerate(tasks):
                    xx, yy = self.data_tasks[t][batch]
                    xx, yy = xx.to(device), yy.to(device)
                    d = state_dicts[i]
                    loss = 0
                    for y in yy.transpose(0,1):
                        im = self.model(xx, list(d.values()))
                        xx = torch.cat([xx[:, im.shape[1]:], im], 1)
                        loss += self.loss_fun(im, y)/yy.shape[1]
                    meta_loss += loss      
                self.meta_optimiser.zero_grad()
                metagrads = torch.autograd.grad(loss, self.weights)
                for w,g in zip(self.weights, metagrads):
                    w.grad=g
                self.meta_optimiser.step()
               
                del meta_loss
                del loss
               # print(torch.cuda.memory_allocated(device))
                
            self.model.eval()
            if (num+1)%32 == 0:
                self.scheduler.step()
                self.inner_lr = self.inner_lr*0.9
                self.meta_lr = self.meta_lr*0.9
                train_mse.append(eval_epoch(train_loaders, self.model, self.loss_fun)[0])
                mse, preds, trues = eval_epoch(valid_loaders, self.model, self.loss_fun)
                valid_mse.append(mse)

                if valid_mse[-1] < min_mse:
                    min_mse = valid_mse[-1] 
                    best_model = self.model
                    torch.save(best_model, name + ".pth")

                end = time.time()
                if (len(train_mse) > 50*16 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
                        break       
                print(num+1, train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(self.meta_optimiser), "5.2e"), name)


def train_epoch_single(data_loader, model, optimizer, loss_function):
    train_mse = []
    k = 0
    for xx, yy in data_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        loss = 0
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, im.shape[1]:], im], 1)
            loss += loss_function(im, y) 
        train_mse.append(loss.item()/yy.shape[1]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse


def eval_epoch_single(data_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in data_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            for y in yy.transpose(0,1):
                im = model(xx)
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