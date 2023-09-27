import torch
import torch.nn as nn
from pad import Model
    
kernel_size = 6  # 패딩 크기 (6 커널)

model = Model(kernel_size)

model.eval()

model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

# fuse하게 되면 모델 구조가 완전히 바뀜
# model_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])

model_prepared = torch.ao.quantization.prepare(model)

input_fp32 = torch.randn(1, 1, 3, 3)
model_prepared(input_fp32)

quantized_model = torch.ao.quantization.convert(model_prepared)


# 모델 양자화 준비
# 원본 모델 저장
torch.save(model, 'model.pth')
torch.jit.save(torch.jit.script(model), 'model_scripted.pth')

print(quantized_model)

# 양자화된 모델 저장
torch.save(quantized_model, 's_quantized_model.pth')
torch.save(quantized_model.state_dict(), 's_quantized_state.pth')
torch.jit.save(torch.jit.script(quantized_model), 's_quantized_model_scripted.pth')

'''
* random values
OrderedDict([('conv.weight', tensor([[[[ 0.0565, -0.2989,  0.0559],
          [ 0.2790, -0.2156,  0.1312],
          [-0.1318,  0.1220,  0.0401]]]])), ('conv.bias', tensor([-0.1458]))])
'''
print(model.state_dict())
'''
* random values
OrderedDict([('quant.scale', tensor([0.0223])), ('quant.zero_point', tensor([74])), ('conv.weight', tensor([[[[ 0.0563, -0.3000,  0.0563],
          [ 0.2789, -0.2156,  0.1313],
          [-0.1313,  0.1219,  0.0398]]]], size=(1, 1, 3, 3), dtype=torch.qint8,
       quantization_scheme=torch.per_channel_affine,
       scale=tensor([0.0023], dtype=torch.float64), zero_point=tensor([0]),
       axis=0)), ('conv.bias', Parameter containing:
tensor([-0.1458], requires_grad=True)), ('conv.scale', tensor(0.0018)), ('conv.zero_point', tensor(0))])
'''
print(quantized_model.state_dict())
