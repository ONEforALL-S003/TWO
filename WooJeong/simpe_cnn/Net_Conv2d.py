import torch
import torch.nn as nn

class Net_Conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 4, 2)
        self.conv2 = nn.Conv2d(4, 8, 2)
        self.conv3 = nn.Conv2d(8, 8, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x