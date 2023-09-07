# Simple PyTorch Model

## 1. Create Single Operator PyTorch Model

```Python
import torch
import torch.nn as nn

class SingleConvModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(SingleConvModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x

# 모델 인스턴스 생성
in_channels = 1  # 입력 이미지의 채널 수 (예: RGB 이미지의 경우 3)
out_channels = 1  # 출력 채널 수
kernel_size = 3  # 컨볼루션 커널 크기 (3x3 커널)

model = SingleConvModel(in_channels, out_channels, kernel_size)
```

- nn.Conv2d를 이용하여 단일 컨볼루션2D 레이어 생성

- 모델의 forward 메서드에서 수행되는 단일 컨볼루션 연산은 입력 데이터를 필터로 합성곱하여 출력을 생성

- forward 메서드는 직접 호출되지 않고, 모델을 입력 데이터에 적용할 때 PyTorch에 의해 자동으로 호출

```python
# 모델 사용 예제
input_tensor = torch.randn(1, in_channels, 3, 3)  # 3x3 크기의 이미지를 입력으로 가정
output_tensor = model(input_tensor)

print(output_tensor.shape)  # 출력 텐서의 크기 출력

# 모델 저장
torch.save(model, "Conv2d.pth")
```

- model안에 인자로 data를 넣으면 알아서 forward 메서드 수행

- torch.save를 통해 모델을 원하는 파일명으로 저장 가능. 보통 .pt, .pth 확장자 사용

## 2. Analyze PyTorch Model

```python
import torch

# 모델 파일 불러오기
model_path = "Conv2d.pth"
loaded_model = torch.load(model_path)

# state_dict 활용한 모델 내부 구조 확인
state_dict = loaded_model.state_dict()
for param_name, param_tensor in state_dict.items():
    print(f"Parameter Name: {param_name}")
    print(f"Parameter Shape: {param_tensor.shape}")
    if 'scale' in param_name or 'zerop' in param_name:
        print(f"Parameter Name: {param_name}")
        print(f"Parameter Value: {param_tensor.item()}")


# 모델 출력
print("model --------------------")
print(loaded_model)
print("model --------------------\n")
# state dict 출력
print("state_dict ---------------")
print(state_dict)
print("state_dict ---------------\n")
```

- 모델 파일을 불러올때는 torch.load를 사용

- state_dict는 PyTorch모델의 파라미터를 저장하는 딕셔너리.
- state_dict의 items() 메소드를 활용하여 내부의 param_name과 param_tensor에 접근할 수 있음
- 분기문은 scale이나 zerop가 있을 경우 출력하는 내용

#### 출력결과

```
Parameter Name: conv.weight
Parameter Shape: torch.Size([1, 1, 3, 3])
Parameter Name: conv.bias
Parameter Shape: torch.Size([1])
model --------------------
SingleConvModel(
  (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1))
)
model --------------------

state_dict ---------------
OrderedDict([('conv.weight', tensor([[[[ 0.1451, -0.1433,  0.0451],
          [-0.2695,  0.1532, -0.1785],
          [ 0.1493, -0.0009, -0.0706]]]])), ('conv.bias', tensor([-0.0027]))])
state_dict ---------------
```

- conv.weight와 conv.bias라는 이름의 파라미터가 내부에 존재

- 모델을 출력하면 SingleConvModel 라는 이름의 사용자 정의 모델 클래스가 나옴.

- (conv) 는 모델 내부의 컨볼루션 레이어를 의미.

- : 뒤의 내용은 conv 레이어의 구조를 나타냄

  - Conv2d : 레이어의 유형
  - 1, 1 : 첫 번째 숫자는 입력 채널 수, 두번째는 출력 채널 수
  - kernel_size=(3,3) : 컨볼루션 커널 크기를 나타냄
  - stride=(1,1) : 스트라이드는 이동간격을 나타냄. (1,1) 은 입력데이터를 슬라이딩할 때 한번씩 이동한다는 뜻.

- state_dict는 OrderedDict를 통해 파라미터 값을 나타냄
  - 'conv.weight' : conv 레이어의 가중치(weight)를 나타냄.
  - 'conv.bias' : conv 레이어의 편향(bias)을 나타냄.

## 3. MobileNetV2 Model

```python
import torchvision.models as models

mobilenetv2_model = models.mobilenetv2.MobileNetV2()

# model information
print("model information -----------------------------")
print(mobilenetv2_model)
print("-----------------------------------------------")
```

- 사전학습된 MobileNetV2 모델을 불러올 수 있음.

#### 출력 결과

```
model information -----------------------------
MobileNetV2(
  (features): Sequential(
    (0): Conv2dNormActivation(
      (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
    (1): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (2): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
          (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )

    ...

    (17): InvertedResidual(
      (conv): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): Conv2dNormActivation(
          (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
          (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (18): Conv2dNormActivation(
      (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
    )
  )
  (classifier): Sequential(
    (0): Dropout(p=0.2, inplace=False)
    (1): Linear(in_features=1280, out_features=1000, bias=True)
  )
)
-----------------------------------------------

```

- features 라는 이름의 Sequantial 레이어로 구성되어 있음.

- 연속된 여러 하위 레이어를 포함하고 있음.
