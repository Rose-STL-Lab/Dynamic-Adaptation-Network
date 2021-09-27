import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import time
from torch.autograd import Variable
from utils import train_epoch, eval_epoch, Dataset, get_lr
from models import DyAd_ResNet, DyAd_Unet
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())

vor_factors = np.load("PhiFlow/task_parameter_vorticity_turbulence.npy")

name = "DyAd"
factors = list(range(1,26))

input_length = 20
out_length = 3
batch_size = 16
num_epoch = 1000
learning_rate = 0.001
min_mse = 1


model = nn.DataParallel(DyAd_ResNet(in_channels= input_length*2, out_channels = 2, kernel_size = 3, style_dim = 512, padding = None, direc = "encoder.pth").to(device))#

test_factors = [3, 8, 13, 18, 23]
factors = list(set(list(range(1,26))) - set(test_factors))

               
direc = "PhiFlow/sliced_data/data"

train_loaders = [(vor_factors[factor-1], data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, direc = direc + str(factor), lst_idx = list(range(0,350))),  batch_size = batch_size, shuffle = True, num_workers = 8)) for factor in factors]

valid_loaders = [(vor_factors[factor-1], data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = out_length, direc = direc + str(factor), lst_idx = list(range(350, 400))), batch_size = batch_size, shuffle = False, num_workers = 8)) for factor in factors]

test_loaders = [(vor_factors[factor-1], data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 20, direc = direc + str(factor), lst_idx = list(range(400, 450))),  batch_size = batch_size, shuffle = False, num_workers = 8)) for factor in factors]

test_loaders2 = [(vor_factors[factor-1], data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 20, direc = direc + str(factor), lst_idx = list(range(50))),  batch_size = batch_size, shuffle = False, num_workers = 8)) for factor in test_factors]


optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()

train_mse = []
valid_mse = []
test_mse = []


for i in range(num_epoch):
    start = time.time()
    scheduler.step()

    model.train()
    train_mse.append(train_epoch(train_loaders, model, optimizer, loss_fun))

    model.eval()
    mse, preds, trues = eval_epoch(valid_loaders, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(model, name + ".pth")

    end = time.time()
    if (len(train_mse) > 100 and np.mean(valid_mse[-10:]) >= np.mean(valid_mse[-20:-10])):
            break       
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)

best_model = torch.load(name + ".pth")
loss_fun = torch.nn.MSELoss()
rmse, preds, trues = eval_epoch(test_loaders, best_model, loss_fun)

rmse2, preds2, trues2 = eval_epoch(test_loaders2, best_model, loss_fun)
torch.save({"future": [rmse, preds, trues],
            "domain": [rmse2, preds2, trues2]}, 
             name + ".pt")
