import torch
import torch.nn as nn
import torch.quantization
# torch.quantization is deprecated. Need to use torch.ao.quantization in latest torch, but we assume to use torch 1.7.0
from Net_Conv2d import Net_Conv2d

torch.manual_seed(123456)

input = torch.randn(4, 2, 4, 6)

model = Net_Conv2d()
model.eval()
state_dict = model.state_dict()
print(model)
tensor_name = state_dict.keys()
model.qconfig = torch.quantization.get_default_qconfig('x86')
p_model = torch.quantization.prepare(model)
p_model(input)
quantized = torch.quantization.convert(p_model)
quant_state_dict = quantized.state_dict()
print(quantized)
quantized_tensor_name = state_dict.keys()
print(tensor_name)
print(quantized_tensor_name)