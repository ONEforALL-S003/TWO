"""
01. Original PyTorch Model
- Operator가 1개인 Conv2d PyTorch Model 생성
- 생성한 모델 내부 Parameter 분석
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.ker = torch.tensor([[[[0.311261, 0.920864, 0.0, 0.0]]]], dtype=torch.float32)
        self.bias = torch.tensor([0.502937], dtype=torch.float32)
        self.layer0 = nn.Conv2d(2, 1, (1, 1), stride=(1, 1), dilation=(1, 1))
        self.layer0.weight = nn.Parameter(self.ker.permute(0, 3, 1, 2))
        self.layer0.bias = nn.Parameter(self.bias)

    def forward(self, ifm):
        ofm = self.layer0(ifm.permute(0, 3, 1, 2))
        return ofm.permute(0, 2, 3, 1)


model = Model()

# ifm, ker, bias 추출
ifm = torch.randn(1, 2, 4, 4)  # 입력 데이터
ker = model.ker
bias = model.bias

# 모델에 입력 데이터 전달하여 ofm 추출
ofm = model(ifm)

# 결과 확인
print("ifm 텐서:")
print(ifm)
"""
ifm 텐서:
tensor([[[[ 1.3207, -1.7929,  0.3953, -0.5921],
          [ 0.9794, -1.3987, -0.4804,  1.1861],
          [-0.8052,  0.0783, -0.3527, -0.8761],
          [-0.8558,  1.5163,  0.2292,  1.3713]],

         [[ 0.3490,  0.0863,  0.0189,  0.7058],
          [-0.1612, -0.6326,  1.2595,  0.3566],
          [ 1.2244, -1.1537, -0.9450, -1.2422],
          [ 0.3794, -0.2434,  0.4723, -1.5399]]]])
"""

print("\nker 텐서:")
print(ker)
"""
ker 텐서:
tensor([[[[0.3113, 0.9209, 0.0000, 0.0000]]]])
"""

print("\nbias 텐서:")
print(bias)
"""
bias 텐서:
tensor([0.5029])
"""

print("\nofm 텐서:")
print(ofm)
"""
ofm 텐서:
tensor([[[[-0.7370],
          [-0.4803],
          [ 0.3244],
          [ 1.6328]],

         [[ 0.6910],
          [-0.1298],
          [-0.1783],
          [ 0.3969]]]], grad_fn=<PermuteBackward0>)
"""

print('#1 print(model)')
print(model)
"""
#1
Model(
  (layer0): Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
)
"""

print('#2 print(model.state_dict())')
print(model.state_dict())
"""
#2 print(model.state_dict())
OrderedDict([('layer0.weight', tensor([[[[0.3113]],

         [[0.9209]],

         [[0.0000]],

         [[0.0000]]]])), ('layer0.bias', tensor([0.5029]))])
"""

print('#3 for layer in model.children()')
for layer in model.children():
    print(layer)
    if isinstance(layer, nn.Conv2d):
        print(layer.state_dict()['weight'])
        print(layer.state_dict()['bias'])
"""
#3 for layer in model.children()
Conv2d(2, 1, kernel_size=(1, 1), stride=(1, 1))
tensor([[[[0.3113]],

         [[0.9209]],

         [[0.0000]],

         [[0.0000]]]])
tensor([0.5029])
"""

print('#4 for name, param in model.named_parameters()')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
        print(param.data)
"""
#4 for name, param in model.named_parameters()
layer0.weight
tensor([[[[0.3113]],

         [[0.9209]],

         [[0.0000]],

         [[0.0000]]]])
layer0.bias
tensor([0.5029])
"""