import torch
import torch.nn as nn
from conv2d import Model

# 모델 생성
model_path = "model.pth"
model = torch.load(model_path)

q_model_path = "s_quantized_model.pth"
# quantized_model = torch.load(q_model_path)
# 양자화된 모델을 불러오기 위해서는 calibration 작업을 생략하고 모델 껍데기만 만들면 된다.
quantized_model = Model(1, 1, 3)
quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
quantized_model = torch.ao.quantization.prepare(quantized_model)
quantized_model = torch.ao.quantization.convert(quantized_model)

# 별도의 변환 작업을 거치지 않으면 형이 맞지 않으므로 제대로 loading 되지 않음
# Copying from quantized Tensor to non-quantized Tensor is not allowed, please use dequantize to get a float Tensor from a quantized Tensor
quantized_model.load_state_dict(torch.load('s_quantized_state.pth'))
model_scripted = torch.jit.load('model_scripted.pth')
quantized_scripted = torch.jit.load('s_quantized_model_scripted.pth')

# 모델 출력
print('Model')
print(model)
print(model_scripted)
print()

print('Quantized')
# AttributeError: 'Conv2d' object has no attribute '_modules'
print(quantized_model) 
print(quantized_scripted)
print()


print(model.state_dict())
print(quantized_model.state_dict())