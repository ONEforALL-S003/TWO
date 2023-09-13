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
* Quantization Parameter 추출
    - [X] torch.quantization.QuantStub()은 모델 변환시 필요할까?

### 2023.09.12
* Extract Quantization Parameter from Single Linear Operation
    - [ ] Specify dtype of tensor