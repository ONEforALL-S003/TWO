# Torch Tester

./nncc test에 의한 자동화에 앞서 Torch의 한 개 짜리 operator를 자동으로 변환, 생성하는 툴을 만든다.

미리 만들어진 한 개 짜리 operator에 대한 변환 결과를 확인할 수 있는 스크립트부터 작성했다.

아직 안정적인 torch-circle 매핑 메서드가 완성되지 않은 관계로 그 전 단계까지만 구현했다.

examples의 모든 모델들은 quantization을 고려하지 않은 모델이므로, `QuantStub`과 `DeQuantStub`으로 앞뒤를 감쌀 수 있는 별도의 wrapper 클래스 안에 넣어서 양자화했다.

## Conv2d

```
PS D:\TWO\Jaehong\tests> python3 .\torch_test.py Conv2d
C:\Users\SSAFY\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\tensorflow_addons\utils\tfa_eol_msg.py:23: UserWarning: 

TensorFlow Addons (TFA) has ended development and introduction of new features.
TFA has entered a minimal maintenance and release mode until a planned end of life in May 2024.
Please modify downstream libraries to take dependencies from other repositories in our TensorFlow community (e.g. Keras, Keras-CV, and Keras-NLP).

For more information see: https://github.com/tensorflow/addons/issues/2807

  warnings.warn(
PyTorch version= 2.0.1+cpu
ONNX version= 1.14.1
ONNX-TF version= 1.10.0
TF version= 2.13.0
Generate 'Conv2d.pth' - Done
C:\Users\SSAFY\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\torch\ao\quantization\observer.py:214: UserWarning: Please use quant_min and quant_max to specify the range for observers.                     reduce_range will be deprecated in a future release of PyTorch.
  warnings.warn(
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