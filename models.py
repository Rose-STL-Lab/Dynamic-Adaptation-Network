import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.nn import Parameter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################## DyAd Encoder ################################
class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x
    
###### Conv3d: time invariant, global mean pool over time and spatial dimension ###########
class encode_layer_3D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size = 3):
        super(encode_layer_3D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv3d(in_dim, out_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2, bias = True),
            nn.BatchNorm3d(out_dim),
            nn.LeakyReLU()
        )
    def forward(self, xx):
        return self.layer(xx)
    
class Encode_Style_3D(nn.Module):
    def __init__(self, in_channels, style_dim = 1024, kernel_size = 3):
        super(Encode_Style_3D, self).__init__()
        self.model = nn.Sequential(
            encode_layer_3D(in_channels, 128),
            nn.MaxPool3d(2),
            encode_layer_3D(128, 256),
            nn.MaxPool3d(2),
            encode_layer_3D(256, 512),
            nn.MaxPool3d(2),
            encode_layer_3D(512, 1024)
        )
        self.linear_c = nn.Linear(style_dim, 1)
        self.linear_z = nn.Linear(1024, style_dim)
        
    def forward(self, xx):
        out = self.model(xx)
        out = torch.mean(out, dim = (2,3,4))
        z = self.linear_z(out)
        return self.linear_c(z), z
    
#################################################################################################################3    
    
## Adaptive Instance Normalization
class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, style_dim, in_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)
        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, xx, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)
        out = self.norm(xx)
        out = gamma * out + beta
        return out
 
## Adaptive Padding
class AdaptivePadding(nn.Module):
    def __init__(self, style_dim, in_channel):
        super().__init__()
        self.padding = nn.Linear(style_dim, 260)
        
    def forward(self, xx, style):
        padding = self.padding(style).unsqueeze(1)
        xx = F.pad(xx, (1,1,1,1))
        xx[:,:,0,0:-1] = padding[:,:,:65]
        xx[:,:,-1,0:-1] = padding[:,:,65:65*2]
        xx[:,:,0:-1,0] = padding[:,:,65*2:65*3]
        xx[:,:,0:-1,-1] = padding[:,:,65*3:65*4]
        return xx

class styleblock(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, padding, style_dim = 512):
        super(styleblock, self).__init__()

        self.layer1 = nn.Conv2d(in_channels, hidden_dim, kernel_size = kernel_size, padding = padding, bias = True)
        self.AdaIN1 = AdaptiveInstanceNorm(style_dim, hidden_dim)
        self.norm1 = nn.ReLU()
        
        self.layer2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = padding, bias = True)
        self.norm2 = nn.ReLU()
        
        self.in_channels = in_channels 
        self.hidden_dim = hidden_dim
        if in_channels != hidden_dim:
            self.upscale = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size = kernel_size, padding = 1, bias = True),
            nn.ReLU()
        )       
        self.AdaPad = AdaptivePadding(style_dim, hidden_dim)
        
    def forward(self, xx, style):
        out = self.norm1(self.layer1(self.AdaPad(xx, style))) # 
        if self.in_channels != self.hidden_dim:
            out =  self.norm2(self.layer2(self.AdaPad(out, style))) + self.upscale(xx)
        else:
            out =  self.norm2(self.layer2(self.AdaPad(out, style))) + xx
        out = self.AdaIN1(out, style)
        return out

######################### DyAd_ResNet ####################################### 

class DyAd_ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding = None, style_dim = 512, direc = None):
        super(DyAd_ResNet, self).__init__()
        if padding == None:
            padding = 0
        model = [styleblock(in_channels, 64, kernel_size, padding), styleblock(64, 64, kernel_size, padding)]
        model += [styleblock(64, 128, kernel_size, padding), styleblock(128, 128, kernel_size, padding)]
        model += [styleblock(128, 256, kernel_size, padding), styleblock(256, 256, kernel_size, padding)] 
        model += [styleblock(256, 512, kernel_size, padding), styleblock(512, 512, kernel_size, padding)]
        model += [nn.Conv2d(512, out_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2, bias = True)]
        self.model = nn.Sequential(*model).to(device)
        
        if not direc:
            print("new encoder")
            self.Style_Gen = Encode_Style_3D(in_channels = 2, style_dim = style_dim).to(device)
        else:
            print("pretrained encoder")
            self.Style_Gen = torch.load(direc).module.to(device)
            for param in self.Style_Gen.parameters():
                param.requires_grad = False

    def forward(self, xx, ss):   
        c, style = self.Style_Gen(ss)
        #style = torch.randn(ss.shape[0], 1024).to(device)
        out = self.model[0](xx, style)
        for block in self.model[1:-1]:
            out = block(out, style)
        return self.model[-1](out)#, c, style
    
    
    
######################### DyAd_Unet #######################################
def conv(input_channels, output_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, stride = stride, padding=(kernel_size - 1) // 2),
        nn.BatchNorm2d(output_channels), 
        nn.LeakyReLU(0.1, inplace = True)
    )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 4, stride = 2, padding=1),
        nn.LeakyReLU(0.1, inplace=True)
    )


class uconv(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, stride, style_dim = 512):
        super(uconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, hidden_dim, kernel_size = kernel_size, padding = 1, stride = stride, bias = True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.AdaIN = AdaptiveInstanceNorm(style_dim, hidden_dim)
        
    def forward(self, xx, style):
        out = self.relu(self.AdaIN(self.conv(xx), style))
        return out     


class DyAd_Unet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, style_dim = 512, direc = None):
        super(DyAd_Unet, self).__init__()
        self.conv1 = uconv(in_channels, 64, kernel_size=kernel_size, stride=2, style_dim = style_dim)
        self.conv1_1 = uconv(64, 64, kernel_size=kernel_size, stride=1, style_dim = style_dim)
        self.conv2 = uconv(64, 128, kernel_size=kernel_size, stride=2, style_dim = style_dim)
        self.conv2_1 = uconv(128, 128, kernel_size=kernel_size, stride=1, style_dim = style_dim)
        self.conv3 = uconv(128, 256, kernel_size=kernel_size, stride=2, style_dim = style_dim)
        self.conv3_1 = uconv(256, 256, kernel_size=kernel_size, stride=1, style_dim = style_dim)
        self.conv4 = uconv(256, 512, kernel_size=kernel_size, stride=2, style_dim = style_dim)
        self.conv4_1 = uconv(512, 1024, kernel_size=kernel_size, stride=1, style_dim = style_dim)

        self.deconv3 = deconv(1024, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
    
        self.output_layer = nn.Conv2d(16 + in_channels, out_channels, kernel_size=kernel_size,
                                      stride = 1, padding=(kernel_size - 1) // 2)
        if not direc:
            print("new encoder")
            self.Style_Gen = Encode_Style_3D(in_channels = 2, style_dim = style_dim).to(device)
        else:
            print("pretrained encoder")
            self.Style_Gen = torch.load(direc).module.to(device)#
            for param in self.Style_Gen.parameters():
                param.requires_grad = False

    def forward(self, xx, ss):   
        c, style = self.Style_Gen(ss)
        
        out_conv1 = self.conv1_1(self.conv1(xx, style), style)
        out_conv2 = self.conv2_1(self.conv2(out_conv1, style), style)
        out_conv3 = self.conv3_1(self.conv3(out_conv2, style), style)
        out_conv4 = self.conv4_1(self.conv4(out_conv3, style), style)

        out_deconv3 = self.deconv3(out_conv4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((xx, out_deconv0), 1)
        out = self.output_layer(concat0)
        return out

