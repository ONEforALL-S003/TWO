import torch
import torch.nn as nn
import torch.quantization
# torch.quantization is deprecated. Need to use torch.ao.quantization in latest torch, but we assume to use torch 1.7.0


class Net_Conv2d_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(8, 8, 2)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(8, 16, 2)
        self.act4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        return x