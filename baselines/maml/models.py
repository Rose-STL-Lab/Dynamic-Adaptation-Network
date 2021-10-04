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
        #w1, b1, m1, s1, rm1, rs1, w2, b2, m2, s2, rm2, rs2 = weights 
        w1, b1, w2, b2 = weights 
       # w1, b1, w2, b2 = w1.to(x.device), b1.to(x.device), w2.to(x.device), b2.to(x.device)#,, m1.to(x.device), s1.to(x.device) m2.to(x.device), s2.to(x.device) 
    
        residual = x
        x = F.pad(x, (1,1,1,1))
        out = F.leaky_relu(F.instance_norm(F.conv2d(x, w1, b1)))#)#, rm1, rs1, m1, s1, momentum=1, training=True))F.batch_norm(F.instance_norm(
        out = F.pad(out, (1,1,1,1))
        out = F.leaky_relu(F.instance_norm(F.conv2d(out, w2, b2)))#F.batch_norm(, rm1, rs1, m2, s2, momentum=1, training=True))
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
            #print(torch.mean(self.model[0].layer1[0].weight).item())
            return out
        else:
            #print(vals[0].shape)
            #weights = {self.keys[i]: vals[i][0].to(xx.device) for i in range(len(self.keys))}
            out = xx
            for i, module in enumerate(self.model_temp):
#                 print(i)
               # print(weights[0].shape, weights[1].shape, weights[2].shape, weights[3].shape, weights[4].shape)
                out = module(out, [weights[i*4+j].to(xx.device) for j in range(4)])
#                 out = module(out, [weights["model."+ str(i) +".layer1.0.weight"], 
#                                    weights["model."+ str(i) +".layer1.0.bias"], 
#                                    weights["model."+ str(i) +".layer1.1.weight"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer1.1.bias"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer1.1.running_mean"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer1.1.running_var"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer2.0.weight"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer2.0.bias"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer2.1.weight"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer2.1.bias"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer2.1.running_mean"],#.to(xx.device),
#                                    weights["model."+ str(i) +".layer2.1.running_var"]])#.to(xx.device)])
            out = F.pad(out, (1,1,1,1))   
            #out = F.conv2d(out, weights['model.11.weight'].to(xx.device), weights['model.11.bias'].to(xx.device))
            out = F.conv2d(out, weights[-2].to(xx.device), weights[-1].to(xx.device))
           # print(weights[0].shape)
            return out
    
        
