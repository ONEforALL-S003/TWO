# Torch Tester

./nncc test에 의한 자동화에 앞서 Torch의 한 개 짜리 operator를 자동으로 변환, 생성하는 툴을 만든다.

미리 만들어진 한 개 짜리 operator에 대한 변환 결과를 확인할 수 있는 스크립트부터 작성했다.

아직 안정적인 torch-circle 매핑 메서드가 완성되지 않은 관계로 그 전 단계까지만 구현했다.

examples의 모든 모델들은 quantization을 고려하지 않은 모델이므로, `QuantStub`과 `DeQuantStub`으로 앞뒤를 감쌀 수 있는 별도의 wrapper 클래스 안에 넣어서 양자화했다.

## Conv2d

```
root@one:/home/ssafy/teamspace/TWO/Jaehong/tests# python3 torch_test.py Conv2d
PyTorch version= 1.7.0
ONNX version= 1.7.0
ONNX-TF version= 1.7.0
TF version= 2.3.0
Generate 'Conv2d.pth' - Done
/usr/local/lib/python3.8/dist-packages/torch/quantization/observer.py:119: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
[W NNPACK.cpp:80] Could not initialize NNPACK! Reason: Unsupported hardware.
Generate 'Conv2d_quantized.pth' - Done
QuantModel(
  (quant): Quantize(scale=tensor([0.0254]), zero_point=tensor([61]), dtype=torch.quint8)
  (net): net_Conv2d(
    (op): QuantizedConv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), scale=0.012533573433756828, zero_point=31)
  )
  (dequant): DeQuantize()
)
quant.scale tensor([0.0254])
quant.zero_point tensor([61])
net.op.weight tensor([[[[ 0.3905]],

         [[-0.1407]]],


        [[[-0.0454]],

         [[ 0.5766]]]], size=(2, 2, 1, 1), dtype=torch.qint8,
       quantization_scheme=torch.per_tensor_affine, scale=0.004540271125733852,
       zero_point=0)
net.op.bias tensor([0.5534, 0.0892], requires_grad=True)
net.op.scale tensor(0.0125)
net.op.zero_point tensor(31)
```

Linear
```
root@one:/home/ssafy/teamspace/TWO/Jaehong/tests# python3 torch_test.py Linear
PyTorch version= 1.7.0
ONNX version= 1.7.0
ONNX-TF version= 1.7.0
TF version= 2.3.0
Generate 'Linear.pth' - Done
/usr/local/lib/python3.8/dist-packages/torch/quantization/observer.py:119: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
Generate 'Linear_quantized.pth' - Done
QuantModel(
  (quant): Quantize(scale=tensor([0.0278]), zero_point=tensor([69]), dtype=torch.quint8)
  (net): net_Linear(
    (op): QuantizedLinear(in_features=3, out_features=6, scale=0.018130801618099213, zero_point=72, qscheme=torch.per_tensor_affine)
  )
  (dequant): DeQuantize()
)
quant.scale tensor([0.0278])
quant.zero_point tensor([69])
net.op.scale tensor(0.0181)
net.op.zero_point tensor(72)
net.op._packed_params.dtype torch.qint8
Traceback (most recent call last):
  File "torch_test.py", line 86, in <module>
    exporter = TorchQParamExporter(quantized_model=quantized, json_path=output_folder + "qparam.json")
  File "/home/ssafy/teamspace/TWO/Jaehong/tests/Torch_QParam_Exporter.py", line 63, in __init__
    self.__extract_module(module=quantized_model)
  File "/home/ssafy/teamspace/TWO/Jaehong/tests/Torch_QParam_Exporter.py", line 154, in __extract_module
    data[tensor_name] = permute(tensor)
  File "/home/ssafy/teamspace/TWO/Jaehong/tests/Torch_QParam_Exporter.py", line 23, in permute
    dim = len(tensor.shape)
AttributeError: 'torch.dtype' object has no attribute 'shape'
```