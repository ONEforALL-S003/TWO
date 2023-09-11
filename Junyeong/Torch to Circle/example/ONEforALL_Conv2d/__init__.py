import torch
import torch.nn as nn

class SingleConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SingleConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x

in_channels = 2
out_channels = 2
kernel_size = 1

_model_ = SingleConvModel(in_channels, out_channels, kernel_size)


#dummy
_dummy_ = torch.randn(1, 2, 3, 3)
