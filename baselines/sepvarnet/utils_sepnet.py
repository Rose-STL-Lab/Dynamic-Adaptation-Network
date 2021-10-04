import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from torch import optim
import numpy as np
import time
import torch.optim.lr_scheduler as lr_scheduler
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def zero_order_loss(s_code_old, s_code_new, skipco = False):
    if skipco:
        s_code_old = torch.cat([s_code_old[0].flatten()] + [x.flatten() for x in s_code_old[1]])
        s_code_new = torch.cat([s_code_new[0].flatten()] + [x.flatten() for x in s_code_new[1]])
    return (s_code_old - s_code_new).pow(2).mean()

def ae_loss(cond, target, sep_net, nt_cond, offset, skipco = False):
    
    """
    Autoencoding function: we consider in this case that:
    if offset == nt_cond:
        St = S_{t, t+1, ..., t+nt_cond} , Tt = T_{t, t+1, ..., t+nt_cond} is associated to t, i.e
        D(St, Tt) = vt,
    somehow like a backward inference.
    if offset == 0:
        St = S_{t, t+1, ..., t+nt_cond} , Tt = T_{t, t+1, ..., t+nt_cond} is associated to t+nt_cond, i.e
        D(St, Tt) = v(t + nt_cond),
    somehow like estimating how dynamic has moved from t up to t + dt
    This function also returns the result of the application of Es on the first and last seen frames.
    """

    full_data = torch.cat([cond, target], dim=1)
    data_new = full_data[:, -nt_cond:]
    data_old = full_data[:, :nt_cond]

    # Encode spatial information
    
    s_code_old = sep_net.Es(data_old, return_skip=skipco)
    s_code_new = sep_net.Es(data_new, return_skip=skipco)

    # Encode motion information at a random time
    if offset == 0:
        t_random = np.random.randint(nt_cond, full_data.size(1))
    else:
        t_random = np.random.randint(nt_cond, full_data.size(1) + 1)
    t_code_random = sep_net.Et(full_data[:, t_random - nt_cond:t_random])

    # Decode from S and random T
    if skipco:
        reconstruction = sep_net.decoder(s_code_old[0], t_code_random, skip=s_code_old[1])
    else:
        reconstruction = sep_net.decoder(s_code_old, t_code_random)

    # AE loss
    supervision_data = full_data[:, t_random - offset]
    loss = F.mse_loss(supervision_data, reconstruction, reduction='mean')

    return loss, s_code_new, s_code_old

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    

class Dataset(data.Dataset):
    def __init__(self, input_length, mid, output_length, direc, lst_idx, stack = False):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc

        self.support_ids = lst_idx
        self.train_ids = lst_idx
        self.stack = stack

    def __len__(self):
        return len(self.train_ids)

    def __getitem__(self, index):
        T_ID = self.train_ids[index]
        y = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        if self.stack:
            x = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1, y.shape[-2], y.shape[-1])
        else:
            x = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[(self.mid-self.input_length):self.mid]
        return x.float(), y.float()
    
def train_epoch(train_loaders, model, optimizer, loss_function, lams):
    train_mse = []
    k = 0
    lamb_ae, lamb_s, lamb_t, lamb_pred = lams
    for data_loader in train_loaders:
        for cond, target in data_loader:
            cond = cond.to(device)
            target = target.to(device)
            total_loss = 0
            optimizer.zero_grad()

            ## AutoEncode
            ae_loss_value, s_recent, s_old = ae_loss(cond, target, model, nt_cond = cond.shape[1], offset = 0)
            total_loss += lamb_ae * ae_loss_value

            # Spatial Invariance
            spatial_ode_loss = zero_order_loss(s_old, s_recent)
            total_loss += lamb_s * spatial_ode_loss

            # Forecast Loss
            full_data = torch.cat([cond, target], dim=1)  # Concatenate all frames
            forecasts, t_codes, _, _ = model.get_forecast(cond, target.shape[1], init_s_code=s_old)
            forecast_loss = F.mse_loss(forecasts, full_data[:, cond.shape[1]:])
            total_loss += lamb_pred * forecast_loss

            # T REGULARIZATION
            t_reg = 0.5 * torch.sum(t_codes[:, 0].pow(2), dim=1).mean()
            total_loss += lamb_t * t_reg

            total_loss.backward()
            optimizer.step()
            train_mse.append(total_loss.item())
        
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loaders, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for data_loader in valid_loaders:
            for cond, target in data_loader:
                cond = cond.to(device)
                target = target.to(device)
                forecasts = model.get_forecast(cond, target.size(1))[0]
                loss = loss_function(forecasts, target)
                preds.append(forecasts.cpu().data.numpy())
                trues.append(target.cpu().data.numpy())
                valid_mse.append(loss.item())
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues
