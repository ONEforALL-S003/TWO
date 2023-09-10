import torch
import torch.nn as nn
from conv2d import Model

in_channels = 1  # 입력 이미지의 채널 수 (예: RGB 이미지의 경우 3)
out_channels = 1  # 출력 채널 수
kernel_size = 3  # 컨볼루션 커널 크기 (3x3 커널)

model = Model(in_channels, out_channels, kernel_size)

# 모델 양자화 준비
# model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 양자화 설정
quantized_model = torch.ao.quantization.quantize_dynamic( # 동적 양자화
    model,  # 원본 모델
    {nn.Conv2d},  # 양자화할 연산자 지정
    dtype=torch.qint8  # 양자화된 가중치와 활성화 함수의 데이터 타입
)

# 원본 모델 저장
torch.save(model, 'model.pth')
torch.jit.save(torch.jit.script(model), 'model_scripted.pth')

print(quantized_model)

# 양자화된 모델 저장
torch.save(quantized_model, 'd_quantized_model.pth')
torch.jit.save(torch.jit.script(quantized_model), 'd_quantized_model_scripted.pth')

# 모델은 완전히 같다.
# 현재 torch의 동적 양자화는 Linear, LSTM 등에만 제한적으로 적용된다.
# 공식문서 참조: https://pytorch.org/docs/stable/quantization.html

