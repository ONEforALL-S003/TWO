import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, kernel_size):
        super(Model, self).__init__()

        self.quant = torch.ao.quantization.QuantStub()
        self.pad = nn.ZeroPad2d(kernel_size)
        self.dequant = torch.ao.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self.pad(x)
        x = self.dequant(x)

        return x