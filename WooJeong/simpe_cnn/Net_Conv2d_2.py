import torch
import torch.nn as nn
import torch.quantization
# torch.quantization is deprecated. Need to use torch.ao.quantization in latest torch, but we assume to use torch 1.7.0


class Net_Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(2, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.conv3 = nn.Conv2d(8, 8, 2)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dequant(x)
        return x
