import torch
import torch.nn as nn
import os.path
import datetime
import numpy as np
import torch
import time
from torch.utils import data
from model import RNN, SpatioTemporalLSTMCell
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count())


name = "PredRNN_TF"
num_layers = 3
num_hidden = [128, 128, 128]
batch_size = 2
input_length = 10
total_length = 20
output_length = total_length - input_length
img_channel = 1
learning_rate = 0.0001

sampling_changing_rate = 0.0002
sampling_stop_iter = 50000
r_exp_alpha = 5000
sampling_start_value = 1

patch_size = 1
img_width = 64
scheduled_sampling = 1
eta = sampling_start_value
def schedule_sampling(eta, itr):
    zeros = np.zeros((batch_size, total_length - input_length - 1, img_width // patch_size, img_width // patch_size, patch_size ** 2 * img_channel))
    if not scheduled_sampling:
        return 0.0, zeros

    if itr < sampling_stop_iter:
        eta -= sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample((batch_size, total_length - input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((img_width // patch_size, img_width // patch_size, patch_size ** 2 * img_channel))
    zeros = np.zeros((img_width // patch_size, img_width // patch_size, patch_size ** 2 * img_channel))
    
    real_input_flag = []
    for i in range(batch_size):
        for j in range(total_length - input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (batch_size, total_length - input_length - 1, img_width // patch_size, img_width // patch_size, patch_size ** 2 * img_channel))
    return eta, real_input_flag


class Dataset(data.Dataset):
    def __init__(self, input_length, mid, output_length, direc, lst_idx, stack = True):
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
        batch = torch.load(self.direc + "/sample_" + str(T_ID) + ".pt")[(self.mid-self.input_length):(self.mid+self.output_length)].permute(0,2,3,1)
        return batch
    
test_factors = [3, 8, 13, 18, 23]
factors = list(set(list(range(1,26))) - set(test_factors))

               
direc = ".../PhiFlow_Buoyancy/data"

train_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = output_length, direc = direc + str(factor), lst_idx = list(range(0,350))),  batch_size = batch_size, shuffle = True, num_workers = 8) for factor in factors]

valid_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = output_length, direc = direc + str(factor), lst_idx = list(range(395, 400))), batch_size = batch_size, shuffle = False, num_workers = 8) for factor in factors]

test_loaders = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 20, direc = direc + str(factor), lst_idx = list(range(400, 450))),  batch_size = batch_size, shuffle = False, num_workers = 8) for factor in factors]

test_loaders2 = [data.DataLoader(Dataset(input_length = input_length, mid = 30, output_length = 20, direc = direc + str(factor), lst_idx = list(range(50))),  batch_size = batch_size, shuffle = False, num_workers = 8) for factor in test_factors]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def train_epoch(train_loaders, model, optimizer):
    train_mse = []
    eta = sampling_start_value
    itr = 0
    for data_loader in train_loaders:
        for batch in data_loader:
            xx = batch.to(device)
            yy = batch[:,input_length+1:].to(device)
            itr += 1
            eta, real_input_flag = schedule_sampling(eta, itr)
            pred_frames, loss = model(xx, torch.FloatTensor(real_input_flag).to(device))
            train_mse.append(loss.item()/yy.shape[1]) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
         
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse

def eval_epoch(valid_loaders, model):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for data_loader in valid_loaders:
            for batch in data_loader:
                
                xx = batch.to(device)
                yy = batch[:,-output_length:].to(device)
                #eta, real_input_flag = schedule_sampling(eta, itr)
                pred_frames, loss = model(xx, xx, test = True)
                preds.append(pred_frames[:, -output_length:].cpu().data.numpy())
                trues.append(yy.cpu().data.numpy())

        preds = np.concatenate(preds, axis = 0)
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean((preds - trues)**2)), 5)
    return valid_mse, preds, trues

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, patch_size = patch_size, img_channel = img_channel, 
                 img_width = img_width, input_length = input_length, total_length = total_length, 
                 reverse_scheduled_sampling = 0):
        super(RNN, self).__init__()
        
        self.input_length = input_length
        self.total_length = total_length
        self.reverse_scheduled_sampling = reverse_scheduled_sampling
        self.frame_channel = patch_size * patch_size * img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = img_width // patch_size
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            
            cell_list.append(SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, 5, 1, 1))
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true, test = False):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()#
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(device)

        for t in range(self.total_length-1):
            # reverse schedule sampling
            if self.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.input_length:
                    net = frames[:, t]
                else:
                    if test:
                        net = x_gen
                    else:
                        #print(mask_true[:batch, t - self.input_length].shape, frames[:, t].shape, x_gen.shape)
                        net = mask_true[:batch, t - self.input_length] * frames[:, t] + \
                              (1 - mask_true[:batch, t - self.input_length]) * x_gen
            #print(net.shape)

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)
            #print(x_gen.shape)
            

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim = 0).permute(1, 0, 3, 4, 2).contiguous()
       # print(next_frames.shape)
        
        loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, loss
    
    
model = RNN(num_layers = num_layers, num_hidden = num_hidden).to(device)

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()

train_mse = []
valid_mse = []
test_mse = []

min_mse = 1
for i in range(1000):
    start = time.time()
    scheduler.step()

    model.train()
    train_mse.append(train_epoch(train_loaders, model, optimizer))

    model.eval()
    rmse, preds, trues = eval_epoch(valid_loaders, model)
    valid_mse.append(rmse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(model, name + ".pth")
    end = time.time()
    if (len(train_mse) > 70 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break       
    print(i+1, train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)

best_model = torch.load(name + ".pth")
loss_fun = torch.nn.MSELoss()
rmse, preds, trues = eval_epoch(test_loaders, best_model)
rmse2, preds2, trues2 = eval_epoch(test_loaders2, best_model)

torch.save({"future": [rmse, preds, trues],
            "domain": [rmse2, preds2, trues2]}, 
             name + ".pt")