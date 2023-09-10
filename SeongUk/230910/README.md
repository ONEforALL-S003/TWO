# PyTorch 모델 → circle 변환 (일)

One의 PyTorchExamples를 이용해서 OneOperModel을 tflite파일로 변환해 보았다.

변환 완료된 tflite파일을 interpreter로 확인해 보았더니
다음과 같았다.

torch에서의 conv2d는 tflite의 convolution;PartitionedCall/convolution 로 변해있었다.



2일차에 진행했던 양자화된 모델을 변환하려고 시도했더니 onnx로 변환화는 과정에서 아래와 같은 에러가 발생하였다.
```
Exception has occurred: RuntimeError
Tried to trace <__torch__.torch.classes.quantized.Conv2dPackedParamsBase object at 0x70567f0> but it is not part of the active trace. Modules that are called during a trace must be registered as submodules of the thing being traced.
  File "/home/ssafy/TWO/SeongUk/230910/day3.py", line 105, in forward
    return self.conv(x)
  File "/home/ssafy/TWO/SeongUk/230910/day3.py", line 68, in convert
    torch.onnx.export(
  File "/home/ssafy/TWO/SeongUk/230910/day3.py", line 117, in <module>
    convert(model_static_quantized, 'quant_OneOperModel')
RuntimeError: Tried to trace <__torch__.torch.classes.quantized.Conv2dPackedParamsBase object at 0x70567f0> but it is not part of the active trace. Modules that are called during a trace must be registered as submodules of the thing being traced.
```
양자화된 operator(QuantizedConv2d)는 지원하지 않는 듯 하다.