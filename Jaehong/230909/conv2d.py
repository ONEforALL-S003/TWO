import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()

        self.quant = torch.ao.quantization.QuantStub()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)

        return x