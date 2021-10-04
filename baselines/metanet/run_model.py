from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import time
from torch.autograd import Variable
from utils import train_epoch, eval_epoch, test_epoch, Dataset, get_lr#, Dataset2
from models import MetaNets
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

name = "MetaNets"
direc = ".../data"
input_length = 25
out_length = 3
batch_size = 8
num_epoch = 1000
learning_rate = 0.001
min_mse = 10

test_factors = [3, 8, 13, 18, 23]
factors = list(set(list(range(1,26))) - set(test_factors))


train_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, direc = direc + str(factor), 
                                         support_idx = list(range(0,350,2)), train_idx = list(range(1,350,2))),  
                                 batch_size = batch_size, shuffle = True, num_workers = 8) for factor in factors]

valid_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, direc = direc + str(factor), 
                                         support_idx = list(range(350, 400, 2)), train_idx = list(range(351, 400, 2))),  
                                 batch_size = batch_size, shuffle = False, num_workers = 8) for factor in factors]

test_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 20, direc = direc + str(factor), 
                                        support_idx = list(range(450, 460)), train_idx = list(range(400, 450))),  
                                 batch_size = batch_size, shuffle = False, num_workers = 8) for factor in factors]

test_loaders2 = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 20, direc = direc + str(factor), 
                                         support_idx = list(range(50, 60)), train_idx = list(range(0, 50))),  
                                 batch_size = batch_size, shuffle = False, num_workers = 8) for factor in test_factors]


model = nn.DataParallel(MetaNets(in_channels = input_length*2, out_channels = 2, hidden_dim = 512, kernel_size = 3).to(device))
print(sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6)




optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.95)
loss_fun = torch.nn.MSELoss()

train_mse = []
valid_mse = []
test_mse = []

for i in range(num_epoch):
    start = time.time()
    scheduler.step()

    train_mse.append(train_epoch(train_loaders, model, optimizer, loss_fun))

    mse, preds, trues = eval_epoch(valid_loaders, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(model, name + ".pth")

    end = time.time()
    if (len(train_mse) > 70 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break       
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)


best_model = torch.load(name + ".pth")
loss_fun = torch.nn.MSELoss()
rmse, preds, trues = eval_epoch(test_loaders, best_model, loss_fun)
rmse2, preds2, trues2 = eval_epoch(test_loaders2, best_model, loss_fun)

torch.save({"future": [rmse, preds, trues],
            "domain": [rmse2, preds2, trues2]}, 
             name + ".pt")

