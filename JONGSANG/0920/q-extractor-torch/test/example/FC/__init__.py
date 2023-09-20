import torch
import torch.nn as nn
import torch.quantization

class Net_FC(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 5)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(5, 10)
        self.act2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(10, 15)
        self.act3 = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        return x

_model_ = Net_FC()

# dummy input for onnx generation
_dummy_ = torch.randn(2, 3)
_qconfig_ = torch.quantization.get_default_qconfig('x86')