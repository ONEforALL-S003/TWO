import torch
from torch import nn
from torch import quantization
    
# Operator가 하나인 모델 생성
class OneOperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

# 모델 생성
model = OneOperModel()

model.eval()
backend = "qnnpack"
model.qconfig = quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = quantization.prepare(model, inplace=False)
model_static_quantized = quantization.convert(model_static_quantized, inplace=False)

# 모델 출력
print('OneOperModel_quant')
print(model_static_quantized)

for key, module in model_static_quantized._modules.items():
    print(module.scale)
    print(module.zero_point)