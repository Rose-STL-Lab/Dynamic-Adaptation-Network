import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.InstanceNorm2d(hidden_dim, affine = False),
            nn.LeakyReLU()
        ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.InstanceNorm2d(hidden_dim, affine = False),
            nn.LeakyReLU()
        ) 
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        
    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(out)
        if self.input_channels == self.hidden_dim:
            out += residual
        return out
    
class Resblock_temp(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Resblock_temp, self).__init__()
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        #self.norm = nn.InstanceNorm2d(hidden_dim, affine=False).to(device)
        #F.batch_norm(x
        
    def forward(self, x, weights):
        w1, b1, w2, b2 = weights 
       
        residual = x
        x = F.pad(x, (1,1,1,1))
        out = F.leaky_relu(F.instance_norm(F.conv2d(x, w1, b1)))
        out = F.pad(out, (1,1,1,1))
        out = F.leaky_relu(F.instance_norm(F.conv2d(out, w2, b2)))
        if self.input_channels == self.hidden_dim:
            out += residual
        return out
    
class MAML_ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(MAML_ResNet, self).__init__()
        layers = [Resblock(input_channels, 64, kernel_size), Resblock(64, 64, kernel_size), Resblock(64, 64, kernel_size)]
        layers += [Resblock(64, 128, kernel_size), Resblock(128, 128, kernel_size), Resblock(128, 128, kernel_size)]
        layers += [Resblock(128, 256, kernel_size), Resblock(256, 256, kernel_size), Resblock(256, 256, kernel_size)]
        layers += [nn.Conv2d(256, output_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)]
        self.model = nn.Sequential(*layers)
        
        self.model_temp = [Resblock_temp(input_channels, 64), Resblock_temp(64, 64) , Resblock_temp(64, 64)]
        self.model_temp += [Resblock_temp(64, 128), Resblock_temp(128, 128), Resblock_temp(128, 128)]
        self.model_temp += [Resblock_temp(128, 256), Resblock_temp(256, 256), Resblock_temp(256, 256)]
        
        self.keys = [key for key, val in self.state_dict().items()]
        
    def clone_state_dict(self):
        cloned_state_dict = {key: val.clone() for key, val in self.state_dict().items()}
        return cloned_state_dict
        
    def forward(self, xx, weights = None):
        if weights == None:
            out = self.model(xx)
            return out
        else:
            out = xx
            for i, module in enumerate(self.model_temp):
                out = module(out, [weights[i*4+j].to(xx.device) for j in range(4)])
            out = F.pad(out, (1,1,1,1))   
            out = F.conv2d(out, weights[-2].to(xx.device), weights[-1].to(xx.device))
            return out
    
        
