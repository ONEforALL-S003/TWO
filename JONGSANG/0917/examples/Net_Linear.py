import torch
import torch.nn as nn

class Net_Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 6)

    def forward(self, x):
        x = self.linear(x)
        return x

_backend_ = 'x86'
_dummy_ = torch.rand(1,2,3,3)