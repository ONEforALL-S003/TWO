import torch
import torch.nn as nn

class PTSQ(nn.Module):
    def __init__(self, model_fp32):
        super(PTSQ, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x