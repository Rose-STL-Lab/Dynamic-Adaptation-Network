import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import time
from utils import train_epoch, eval_epoch, Dataset, get_lr, MAML_train, train_epoch_single, eval_epoch_single
from models import MAML_ResNet
import copy
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())
torch.multiprocessing.set_sharing_strategy('file_system')

name = "maml"

input_length = 20
out_length = 4
batch_size = 8
num_epoch = 2000
learning_rate = 0.001
min_mse = 10


meta_lr = 0.0001
inner_lr = 0.0001
num_per_batch = 10
min_mse = 1


test_factors = [3, 8, 13, 18, 23]
factors = list(set(list(range(1,26))) - set(test_factors))

direc = ".../data"

train_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, direc = direc + str(factor), 
                                         lst_idx = list(range(0,350))),  
                                 batch_size = batch_size, shuffle = True, num_workers = 8) for factor in factors]

valid_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, direc = direc + str(factor), 
                                         lst_idx = list(range(350, 400))), 
                                 batch_size = batch_size, shuffle = False, num_workers = 8) for factor in factors]

test_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 10, direc = direc + str(factor), 
                                        lst_idx = list(range(400, 450))), 
                                batch_size = batch_size, shuffle = False, num_workers = 8) for factor in factors]

test_loaders2 = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 10, direc = direc + str(factor), 
                                         lst_idx = list(range(50))),  
                                 batch_size = batch_size, shuffle = False, num_workers = 8) for factor in test_factors]

adapt_test_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 10, direc = direc + str(factor), 
                                        lst_idx = list(range(450, 460))), 
                                batch_size = batch_size, shuffle = False, num_workers = 8) for factor in factors]

adapt_test_loaders2 = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 10, direc = direc + str(factor), 
                                         lst_idx = list(range(50,60))),  
                                 batch_size = batch_size, shuffle = False, num_workers = 8) for factor in test_factors]



data_tasks = [list(task_loader) for task_loader in train_loaders]


model = MAML_ResNet(input_channels= input_length, output_channels = 1, kernel_size = 3).to(device)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

meta_optimiser = torch.optim.Adam(model.parameters(), meta_lr, betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, step_size= 1, gamma=0.95)
loss_fun = torch.nn.MSELoss()

maml = MAML_train(data_tasks, model, meta_optimiser, scheduler, num_per_batch, inner_lr, meta_lr)
maml.main_loop(num_epoch, train_loaders, valid_loaders, name)


# 
preds, trues = [], []
loss_fun = torch.nn.MSELoss()
for t in range(len(adapt_test_loaders)):
    model = torch.load(name + ".pth")
    optimiser = torch.optim.Adam(model.parameters(), 1e-5, betas=(0.9, 0.999), weight_decay=4e-4)
    model.train()
    for i in range(10):
        _ = train_epoch_single(adapt_test_loaders[t], model, optimiser, loss_fun)
    model.eval()
    valid_mse, pred, true = eval_epoch_single(test_loaders[t], model,  loss_fun) 
    preds.append(pred)
    trues.append(true)
preds, trues = np.concatenate(preds, axis = 0), np.concatenate(trues, axis = 0)
future_results = [np.sqrt(np.mean((preds - trues)**2)), preds[::10], trues[::10]]

preds, trues = [], []
for t in range(len(adapt_test_loaders2)):
    model = torch.load(name + ".pth")
    optimiser = torch.optim.Adam(model.parameters(), 1e-5, betas=(0.9, 0.999), weight_decay=4e-4)
    model.train()
    for i in range(10):
        _= train_epoch_single(adapt_test_loaders2[t], model, optimiser, loss_fun)
    model.eval()
    valid_mse, pred, true = eval_epoch_single(test_loaders2[t], model,  loss_fun)  
    preds.append(pred)
    trues.append(true)
preds, trues = np.concatenate(preds, axis = 0), np.concatenate(trues, axis = 0)
domain_results = [np.sqrt(np.mean((preds - trues)**2)), preds[::10], trues[::10]]

torch.save({"future": future_results,
            "domain": domain_results}, 
             name + ".pt")
