# TIL

우선 `torch`에서 `Conv2D`는 dynamic quantization이 지원하지 않는다. 실제로 동적 양자화를 한 후에도 아무 변화가 없다.

정적으로 양자화하면 `Conv2d` 클래스가 `QuantizedConv2d`로 바뀌어 들어간다.

```py
'''
OrderedDict([('conv.weight', tensor([[[[ 0.0565, -0.2989,  0.0559],
          [ 0.2790, -0.2156,  0.1312],
          [-0.1318,  0.1220,  0.0401]]]])), ('conv.bias', tensor([-0.1458]))])
'''
print(model.state_dict())
'''
OrderedDict([('quant.scale', tensor([0.0223])), ('quant.zero_point', tensor([74])), ('conv.weight', tensor([[[[ 0.0563, -0.3000,  0.0563],
          [ 0.2789, -0.2156,  0.1313],
          [-0.1313,  0.1219,  0.0398]]]], size=(1, 1, 3, 3), dtype=torch.qint8,
       quantization_scheme=torch.per_channel_affine,
       scale=tensor([0.0023], dtype=torch.float64), zero_point=tensor([0]),
       axis=0)), ('conv.bias', Parameter containing:
tensor([-0.1458], requires_grad=True)), ('conv.scale', tensor(0.0018)), ('conv.zero_point', tensor(0))])
'''
print(quantized_model.state_dict())
```

state_dict로 확인하면 양자화된 모델에서도 float으로 변환된 값을 바로 확인할 수 있다. 값을 저장하면 텐서 값은 정수로 들어가있다.

한 가지 분명한 점은 torch 모델을 불러오기 위해서는 Model이 어딘가에 선언되어 있어야 한다는 것이다.

여기서 하나 더 문제가 생기는데, 양자화된 모델을 불러오기 위한 적절한 생성자가 없다는 것이다.

따라서 양자화된 모델의 가중치를 로딩하기 전에 껍데기만 적당히 변환해주는 작업이 필요하다.

```py
# 양자화된 모델을 불러오기 위해서는 calibration 작업을 생략하고 모델 껍데기만 만들면 된다.
quantized_model = Model(1, 1, 3)
quantized_model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
quantized_model = torch.ao.quantization.prepare(quantized_model)
quantized_model = torch.ao.quantization.convert(quantized_model)

quantized_model.load_state_dict(torch.load('s_quantized_state.pth'))
```

`weight(tensor)`가 자체적으로 갖고 있는 qparams는 양자화를 위한 것으로 보인다.
`conv(quantizedConv2d)`가 갖고 있는 qparams는 output을 역양자화할 때 쓰는 것인지 모르겠다.
