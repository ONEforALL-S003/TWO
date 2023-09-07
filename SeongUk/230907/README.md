# Original Torch Model (목)

## 1. Operator가 1개인 PyTorch Model 생성

``` python
# Operator가 하나인 모델 생성
class OneOperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

model = OneOperModel()
```

모델 생성 부분은 pythorch의 공식문서를 참고하여 작성하였다.   
[참고한 문서](https://pytorch.org/docs/stable/quantization.html#quantization-flow)



## 2. 생성한 모델 내부 Parameter 분석

``` python
# 내부 구조 출력 함수
def torch_detail(model):
    for key, module in model._modules.items():
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        print('(' + key + '): ' + mod_str)

# 모델 출력
torch_detail(model)
```

pytorch는 주로 OrderdDict 클래스를 이용하여 모델을 읽어온다.

출력 양식은 pytorch model class 인 torch.nn.Module 의 내부 메서드인 __repr__을 참고하여 작성 하였다.

