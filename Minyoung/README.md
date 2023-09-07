# 정민영

## operator가 1개인 모델에 대해서 전반적 프로세스 진행

### 01. Original PyTorch Model
    - Operator가 1개인 PyTorch Model 생성
    - 생성한 모델 내부 Parameter 분석

### 02. PyTorch 모델 Quantization
    - Quantized Torch Model 분석

### 03. PyTorch 모델 → circle 변환
    - torch → ONNX → tensorflow → tensorflow lite → circle 모델로 변환
    - circle 모델 분석

### 04. q-implant
    - q-param 정의
    - circle 모델에 q-param implant

### 05. Quantized circle model
    - quantized circle model 확인