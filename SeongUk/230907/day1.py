import torch
from torch import nn

# Operator가 하나인 모델 생성
class OneOperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s

# 내부 구조 출력 함수
def torch_detail(model):
    for key, module in model._modules.items():
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        print('(' + key + '): ' + mod_str)

# 모델 생성
model = OneOperModel()

# 모델 출력
print('OneOperModel')
torch_detail(model)
print()