# Torch Tester

./nncc test에 의한 자동화에 앞서 Torch의 한 개 짜리 operator를 자동으로 변환, 생성하는 툴을 만든다.

미리 만들어진 한 개 짜리 operator에 대한 변환 결과를 확인할 수 있는 스크립트부터 작성했다.

아직 안정적인 torch-circle 매핑 메서드가 완성되지 않은 관계로 그 전 단계까지만 구현했다.

examples의 모든 모델들은 quantization을 고려하지 않은 모델이므로, `QuantStub`과 `DeQuantStub`으로 앞뒤를 감쌀 수 있는 별도의 wrapper 클래스 안에 넣어서 양자화했다.

## Conv2d

```
PS D:\TWO\Jaehong\tests> python3 .\torch_test.py Conv2d
PyTorch version= 2.0.1+cpu
ONNX version= 1.14.1
ONNX-TF version= 1.10.0
TF version= 2.13.0
Generate 'Conv2d.pth' - Done
Generate 'Conv2d_quantized.pth' - Done
QuantModel(
  (quant): Quantize(scale=tensor([0.0221]), zero_point=tensor([80]), dtype=torch.quint8)
  (net): net_Conv2d(
    (op): QuantizedConv2d(2, 2, kernel_size=(1, 1), stride=(1, 1), scale=0.013211706653237343, zero_point=28)
  )
  (dequant): DeQuantize()
)
quant.scale tensor([0.0221])
quant.zero_point tensor([80])
net.op.weight tensor([[[[-0.0509]],

         [[-0.4649]]],


        [[[-0.5997]],

         [[-0.4592]]]], size=(2, 2, 1, 1), dtype=torch.qint8,
       quantization_scheme=torch.per_channel_affine,
       scale=tensor([0.0036, 0.0047], dtype=torch.float64),
       zero_point=tensor([0, 0]), axis=0)
Traceback (most recent call last):
  File "D:\TWO\Jaehong\tests\torch_test.py", line 86, in <module>
    exporter = TorchQParamExporter(quantized_model=quantized, json_path=output_folder + "qparam.json")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\TWO\Jaehong\tests\Torch_QParam_Exporter.py", line 63, in __init__
    self.__extract_module(module=quantized_model)
  File "D:\TWO\Jaehong\tests\Torch_QParam_Exporter.py", line 154, in __extract_module
    data[tensor_name] = permute(tensor)
                        ^^^^^^^^^^^^^^^
  File "D:\TWO\Jaehong\tests\Torch_QParam_Exporter.py", line 25, in permute
    tensor = tensor.permute(0, 2, 3, 1)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Setting strides is possible only on uniformly quantized tensor
```

Linear
```
PS D:\TWO\Jaehong\tests> python3 .\torch_test.py Linear
PyTorch version= 2.0.1+cpu
ONNX version= 1.14.1
ONNX-TF version= 1.10.0
TF version= 2.13.0
Generate 'Linear.pth' - Done
Generate 'Linear_quantized.pth' - Done
QuantModel(
  (quant): Quantize(scale=tensor([0.0385]), zero_point=tensor([58]), dtype=torch.quint8)
  (net): net_Linear(
    (op): QuantizedLinear(in_features=3, out_features=6, scale=0.03507407754659653, zero_point=49, qscheme=torch.per_channel_affine)
  )
  (dequant): DeQuantize()
)
quant.scale tensor([0.0385])
quant.zero_point tensor([58])
net.op.scale tensor(0.0351)
net.op.zero_point tensor(49)
net.op._packed_params.dtype torch.qint8
Traceback (most recent call last):
  File "D:\TWO\Jaehong\tests\torch_test.py", line 86, in <module>
    exporter = TorchQParamExporter(quantized_model=quantized, json_path=output_folder + "qparam.json")
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\TWO\Jaehong\tests\Torch_QParam_Exporter.py", line 63, in __init__
    self.__extract_module(module=quantized_model)
  File "D:\TWO\Jaehong\tests\Torch_QParam_Exporter.py", line 154, in __extract_module
    data[tensor_name] = permute(tensor)
                        ^^^^^^^^^^^^^^^
  File "D:\TWO\Jaehong\tests\Torch_QParam_Exporter.py", line 23, in permute
    dim = len(tensor.shape)
              ^^^^^^^^^^^^
AttributeError: 'torch.dtype' object has no attribute 'shape'
```