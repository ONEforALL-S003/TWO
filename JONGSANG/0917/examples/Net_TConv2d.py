import torch
import torch.nn as nn

class Net_TConv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.TConv2d = nn.ConvTranspose2d(6,3,2)
    
    def forward(self, x):
        x = self.TConv2d(x)
        return x

_backend_ = 'qnnpack'
_dummy_ = torch.rand(1,6,3,3)