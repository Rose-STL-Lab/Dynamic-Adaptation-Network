import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Task_Module(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim, kernel_size):
        super(Task_Module, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim, output_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)
        ) 
       
    def forward(self, xx):
        out = self.model(xx)
        return out
    
class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
    

class ModularMeta(nn.Module):
    def __init__(self, input_channels, output_channels, hidden_dim, kernel_size, num_tasks, attention = False, weight_attn = False):
        super(ModularMeta, self).__init__()
        self.module_list = nn.ModuleList([Task_Module(input_channels, output_channels, hidden_dim, kernel_size).to(device) for i in range(num_tasks)])
        self.attention = attention
        self.weight_attn = weight_attn
        if self.attention:
            self.attn = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size = kernel_size, padding = (kernel_size-1)//2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size = kernel_size, padding = (kernel_size-1)//2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size = kernel_size, padding = (kernel_size-1)//2),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 64, kernel_size = kernel_size, padding = (kernel_size-1)//2),
                Flatten(),
                nn.Linear(4096, num_tasks)
            ) 
       
    def forward(self, xx, task_id):
        if self.attention:
            if not self.weight_attn:
                f = F.softmax(self.attn(xx))
                outputs = torch.cat([m(xx).unsqueeze(-1) for m in self.module_list], dim = -1)
                out = torch.einsum("abcde, ae -> abcd", outputs, f)
            else:
                outs = []
                f = F.softmax(self.attn(xx))
                outputs = [torch.cat([list(m.parameters())[l].clone().unsqueeze(-1) for m in self.module_list], dim = -1) for l in range(14)]
                for s in range(xx.shape[0]): 
                    inter_w1 = torch.sum(outputs[0]*f[s:s+1].unsqueeze(1).unsqueeze(1).unsqueeze(1), dim = -1)
                    inter_b1 = torch.sum(outputs[1]*f[s:s+1], dim = -1) 
                    out = F.leaky_relu(F.conv2d(F.pad(xx[s:s+1], (1,1,1,1)), inter_w1, inter_b1))
                    
                    inter_w2 = torch.sum(outputs[4]*f[s:s+1].unsqueeze(1).unsqueeze(1).unsqueeze(1), dim = -1)
                    inter_b2 = torch.sum(outputs[5]*f[s:s+1], dim = -1) 
                    out = F.leaky_relu(F.conv2d(F.pad(out, (1,1,1,1)), inter_w2, inter_b2))
                    
                    inter_w3 = torch.sum(outputs[8]*f[s:s+1].unsqueeze(1).unsqueeze(1).unsqueeze(1), dim = -1)
                    inter_b3 = torch.sum(outputs[9]*f[s:s+1], dim = -1) 
                    out = F.leaky_relu(F.conv2d(F.pad(out, (1,1,1,1)), inter_w3, inter_b3))
                    
                    inter_w4 = torch.sum(outputs[12]*f[s:s+1].unsqueeze(1).unsqueeze(1).unsqueeze(1), dim = -1)
                    inter_b4 = torch.sum(outputs[13]*f[s:s+1], dim = -1) 
                    out = F.conv2d(F.pad(out, (1,1,1,1)), inter_w4, inter_b4)
                    outs.append(out)
                outs = torch.cat(outs, dim = 0)
                return outs
        else:
            out = self.module_list[task_id](xx)
        return out