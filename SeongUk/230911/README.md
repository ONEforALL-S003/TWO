# q-implant (월)

### q-param 정의

```
OneOperModel_quant
OneOperModel(
  (quant): Quantize(scale=tensor([1.]), zero_point=tensor([0]), dtype=torch.quint8)
  (conv): QuantizedConv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
  (dequant): DeQuantize()
)
(tensor([[[[0.9467]]]], size=(1, 1, 1, 1), dtype=torch.qint8,
       quantization_scheme=torch.per_tensor_affine, scale=0.007453964091837406,
       zero_point=0),
 tensor([-0.2587], requires_grad=True))
<class 'tuple'>
<class 'torch.Tensor'>
<class 'torch.Tensor'>
```

추출한 weight와 bias의 클래스가 torch.Tensor여서 torch.Tensor의 소스코드 중 deepcopy부분을 이용하여 scale과 zeropoint를 확인할 수 있었다.
또한 텐서의 qscheme 가 torch.per_channel_affine, torch.per_channel_affine_float_qparams 중 하나라면
axis라는 항목을 추가로 복사하는데 이 부분이 quantized_dimension이 아닌가 싶다.