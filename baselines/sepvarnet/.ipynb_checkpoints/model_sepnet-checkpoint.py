import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from torch import optim
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def activation_factory(name):
    """
    Returns the activation layer corresponding to the input activation name.
    Parameters
    ----------
    name : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', or 'tanh'. Adds the corresponding activation function after the
        convolution.
    """
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    if name is None or name == "identity":
        return nn.Identity()

    raise ValueError(f'Activation function `{name}` not yet implemented')
    
def make_conv_block(conv, activation, bn=True):
    """
    Supplements a convolutional block with activation functions and batch normalization.
    Parameters
    ----------
    conv : torch.nn.Module
        Convolutional block.
    activation : str
        'relu', 'leaky_relu', 'elu', 'sigmoid', 'tanh', or 'none'. Adds the corresponding activation function after
        the convolution.
    bn : bool
        Whether to add batch normalization after the activation.
    """
    out_channels = conv.out_channels
    modules = [conv]
    if bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if activation != 'none':
        modules.append(activation_factory(activation))
    return nn.Sequential(*modules)

class EncoderSST(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncoderSST, self).__init__()
        self.conv1 = nn.Sequential(
                make_conv_block(nn.Conv2d(in_c, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu'))
        self.conv2 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(64, 64 * 2, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 2, 64 * 2, 3, 1, 1), activation='leaky_relu'),
            )
        self.conv3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                make_conv_block(nn.Conv2d(64 * 2, 64 * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 4, 64 * 4, 3, 1, 1), activation='leaky_relu'),
            )
        self.conv4 = nn.Sequential(
                make_conv_block(nn.Conv2d(64 * 4, 64 * 8, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64 * 8, out_c, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(out_c, out_c, 3, 1, 1), activation='none', bn=False),
            )

    def forward(self, x, return_skip=False):
        x = x.view(x.size(0), -1, x.size(3), x.size(4))
        # skipcos
        h1 = self.conv1(x)  # 64, 64, 64
        h2 = self.conv2(h1)  # 128, 32, 32
        h3 = self.conv3(h2)  # 256, 16, 16
        # code
        h4 = self.conv4(h3)  # 512, 16, 16
        if return_skip:
            return h4, [h3, h2, h1]
        return h4

class DecoderSST(nn.Module):
    def __init__(self, in_c, out_c):
        super(DecoderSST, self).__init__()
        self.conv1 = nn.Sequential(
                make_conv_block(nn.Conv2d(in_c, 256, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(256, 256, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(256, 128, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest')
        )  # 128, 32, 32

        self.conv2 = nn.Sequential(
                make_conv_block(nn.Conv2d(128, 128, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(128, 128, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(128, 64, 3, 1, 1), activation='leaky_relu'),
                nn.Upsample(scale_factor=2, mode='nearest'),
        )  # 64, 64, 64
        self.conv3 = nn.Sequential(
                make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu'),
                make_conv_block(nn.Conv2d(64, out_c, 3, 1, 1), activation='leaky_relu'))

        #self.out_f = activation_factory(out_f)

    def forward(self, s_code, t_code, skip=None):
        x = torch.cat([s_code, t_code], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
  # return self.out_f(x)
class ConvResBlock(nn.Module):
    def __init__(self, in_c, out_c, nf=64):
        super(ConvResBlock, self).__init__()
        self.conv = nn.Sequential(
            make_conv_block(nn.Conv2d(in_c, nf, 3, padding=1), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf, nf, 3, padding=1), activation='leaky_relu'),
            make_conv_block(nn.Conv2d(nf, out_c, 3, padding=1), activation='none')
        )
        self.in_c = in_c
        self.out_c = out_c
        if in_c != out_c:
            # conv -- bn -- leaky_relu
            self.up = make_conv_block(nn.Conv2d(in_c, out_c, 3, padding=1), activation='none')
#         else:
#             self.up = nn.Identity()

    def forward(self, x):
        residual = self.conv(x)
        if self.in_c != self.out_c:
            x = self.up(x) + residual
        else:
            x = x + residual
        return x, residual


class ConvResnet(nn.Module):
    def __init__(self, in_c, n_blocks=1, nf=64):
        super(ConvResnet, self).__init__()
        self.n_blocks = n_blocks
        self.resblock_modules = nn.ModuleList()
        for i in range(n_blocks):
            self.resblock_modules.append(ConvResBlock(in_c, in_c, nf=nf))

    def forward(self, x, return_res=True):
        residuals = []
        for i, residual_block in enumerate(self.resblock_modules):
            x, residual = residual_block(x)
            residuals.append(residual)
        if return_res:
            return x, residuals
        return x
        
class SeparableNetwork(nn.Module):

    def __init__(self, Es, Et, t_resnet, decoder, nt_cond, skipco = False):
        super(SeparableNetwork, self).__init__()

        assert isinstance(Es, nn.Module)
        assert isinstance(Et, nn.Module)
        assert isinstance(t_resnet, nn.Module)
        assert isinstance(decoder, nn.Module)

        # Networks
        self.Es = Es
        self.Et = Et
        self.decoder = decoder
        self.t_resnet = t_resnet

        # Attributes
        self.nt_cond = nt_cond
        self.skipco = skipco

        # Gradient-enabling parameter
        self.__grad = True

    @property
    def grad(self):
        return self.__grad

    @grad.setter
    def grad(self, grad):
        assert isinstance(grad, bool)
        self.__grad = grad

    def get_forecast(self, cond, n_forecast, init_t_code=None, init_s_code=None):
        t_codes = []
        forecasts = []
        t_residuals = []

        if init_s_code is None:
            s_code = self.Es(cond, return_skip=self.skipco)
        else:
            s_code = init_s_code
        if self.skipco:
            s_code, s_skipco = s_code
        else:
            s_skipco = None

        if init_t_code is None:
            t_code = self.Et(cond)
        else:
            t_code = init_t_code

        t_codes.append(t_code)

        # Decode first frame
        forecast = self.decoder(s_code, t_code, skip=s_skipco)
        forecasts.append(forecast)

        # Forward prediction
        for t in range(1, n_forecast):
            #print(t_code.shape)
            t_code, t_res = self.t_resnet(t_code)
            t_codes.append(t_code)
            t_residuals.append(t_res)
            forecast = self.decoder(s_code, t_code, skip=s_skipco)
            forecasts.append(forecast)

        # Stack predictions
        forecasts = torch.cat([x.unsqueeze(1) for x in forecasts], dim=1)
        t_codes = torch.cat([x.unsqueeze(1) for x in t_codes], dim=1)

        return forecasts, t_codes, s_code, t_residuals