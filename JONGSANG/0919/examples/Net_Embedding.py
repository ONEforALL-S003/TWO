import torch
import torch.nn as nn

## fail to quantize
class Net_Embedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10,12)

    def forward(self,x):
        x = self.embedding(x)
        return x