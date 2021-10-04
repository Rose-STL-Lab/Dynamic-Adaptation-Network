import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from torch import optim
import numpy as np
import time
from model_sepnet import EncoderSST, ConvResnet, DecoderSST, SeparableNetwork
from utils_sepnet import train_epoch, eval_epoch, Dataset, get_lr
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

code_size_t = 64
code_size_s = 192
num_epoch = 3000
nc = 2
nt_cond = 20
nt_pred = 6
offset = 0
lr = 0.001
batch_size = 16
lamb_ae = 10
lamb_s = 100
lamb_t = 1e-6
lamb_pred = 50

direc = ".../data"

name = "sep_net_bz{}_cond{}_pred{}_lr{}_ae{}_s{}_t{}_pr{}".format(batch_size, nt_cond, nt_pred, lr, lamb_ae, lamb_s, lamb_t, lamb_pred)
test_factors = [3, 8, 13, 18, 23]
factors = list(set(list(range(1,26))) - set(test_factors))
# test_factors = [3, 7, 12, 20, 22]
# factors = list(set(list(range(1,26))) - set(test_factors))

train_set = Dataset(input_length = nt_cond, mid = 30, output_length = nt_pred, direc = direc, task_list = factors, sample_list = list(range(350)))
valid_set = Dataset(input_length = nt_cond, mid = 30, output_length = nt_pred, direc = direc, task_list = factors, sample_list = list(range(350, 400)))
test_set_time = Dataset(input_length = nt_cond, mid = 30, output_length = 20, direc = direc, task_list = factors, sample_list = list(range(400, 450)))
test_set_domain = Dataset(input_length = nt_cond, mid = 30, output_length = 20, direc = direc, task_list = test_factors, sample_list = list(range(50)))

train_loader = data.DataLoader(train_set,  batch_size = batch_size, shuffle = True, num_workers = 8)       
valid_loader = data.DataLoader(valid_set,  batch_size = batch_size, shuffle = True, num_workers = 8)            
test_loader_time = data.DataLoader(test_set_time,  batch_size = batch_size, shuffle = False, num_workers = 8)            
test_loader_domain = data.DataLoader(test_set_domain,  batch_size = batch_size, shuffle = False, num_workers = 8)  

Et = EncoderSST(in_c = nt_cond*nc, out_c = code_size_t).to(device)
Es = EncoderSST(in_c = nt_cond*nc, out_c = code_size_s).to(device)
t_resnet = ConvResnet(in_c = 64, n_blocks = 2).to(device)
decoder = DecoderSST(in_c = code_size_t + code_size_s, out_c = nc).to(device)
sep_net = SeparableNetwork(Es, Et, t_resnet, decoder, nt_cond).to(device)
print(sum(p.numel() for p in sep_net.parameters() if p.requires_grad)/1e6)

optimizer = optim.Adam(sep_net.parameters(), lr=lr, betas=(0.9, 0.99))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()


train_mse = []
valid_mse = []
test_mse = []
min_mse = 1

for i in range(num_epoch):
    start = time.time()
    scheduler.step()

    sep_net.train()
    train_mse.append(train_epoch(train_loader, sep_net, optimizer, loss_fun, lams = [lamb_ae, lamb_s, lamb_t, lamb_pred]))

    sep_net.eval()
    mse, preds, trues = eval_epoch(valid_loader, sep_net, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = sep_net
        torch.save(sep_net, name + ".pth")

    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break       
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)

best_model = torch.load(name + ".pth")
loss_fun = torch.nn.MSELoss()


rmse, preds, trues = eval_epoch(test_loader_time, best_model, loss_fun)
rmse2, preds2, trues2 = eval_epoch(test_loader_domain, best_model, loss_fun)

torch.save({"future": [rmse, preds, trues],
            "domain": [rmse2, preds2, trues2]}, 
             name + ".pt")