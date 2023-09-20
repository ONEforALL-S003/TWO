# ONE for ALL

### 2023.09.07
* Generate Single Linear Operation (PyTorch)
    * ONE/res/PyTorchExamples/examples/Linear
* Single Linear Operation (PyTorch -> Circle)

### 2023.09.10
* Quantize Single Linear Operation

### 2023.09.11
* Convert PyTorch to Circle (fp32 model == not quantized)
    - [X] TFLite -> Circle 변환시 가중치가 잘못 변환되는 문제
          - Netron에서 잘못된 파일로 확인해서 벌어진 문제
* Quantization Parameter 추출
    - [X] torch.quantization.QuantStub()은 모델 변환시 필요할까?

### 2023.09.12
* Extract Quantization Parameter from Single Linear Operation
    - [ ] Specify dtype of tensor

### 2023.09.13
* Extract Quantization Parameter from _packed_params

### 2023.09.19
* Extract Quantization Parameter from Single ConvTranspose2d Operation
    - [ ] Not mapped QP exists

### 2023.09.20
* Q-extractor-torch를 테스트하기 위한, 사전 준비
    * one/res/PyTorchExamples/examples/의 모든 OP를 PTSQ
