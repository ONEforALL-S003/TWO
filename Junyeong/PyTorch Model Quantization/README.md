# PyTorch Model Quantization

## 1. Quantize Single Operator PyTorch Model

-> Conv2dModelQuantization.ipynb 참조

## 2.Analyze

### 동적 양자화

#### 원본 모델, state_dict

```
SingleConvModel(
  (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1))
)
OrderedDict([('conv.weight', tensor([[[[-0.1645,  0.2326,  0.1386],
          [-0.2683,  0.2556,  0.0943],
          [-0.2318,  0.0654,  0.0817]]]])), ('conv.bias', tensor([-0.1088]))])
```

#### 동적양자화 이후 모델, state_dict

```
SingleConvModel(
  (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1))
)
OrderedDict([('conv.weight', tensor([[[[-0.1645,  0.2326,  0.1386],
          [-0.2683,  0.2556,  0.0943],
          [-0.2318,  0.0654,  0.0817]]]])), ('conv.bias', tensor([-0.1088]))])
```

동적 양자화에서는 차이가 없다.

### 정적 양자화

정적 양자화를 진행하려면 QuantStub, DeQuantStub, ReLU를 추가해 주는 것으로 보임

#### 원본 모델, state_dict

```
M(
  (quant): QuantStub()
  (conv): Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1))
  (relu): ReLU()
  (dequant): DeQuantStub()
)
OrderedDict([('conv.weight', tensor([[[[-0.2928,  0.0156,  0.2505],
          [ 0.1452,  0.1279,  0.3321],
          [ 0.0904,  0.2869,  0.1864]]]])), ('conv.bias', tensor([-0.0878]))])
```

#### 정적양자화 이후 모델, state_dict

```
M(
  (quant): Quantize(scale=tensor([0.0430]), zero_point=tensor([63]), dtype=torch.quint8)
  (conv): QuantizedConvReLU2d(1, 1, kernel_size=(3, 3), stride=(1, 1), scale=0.008406509645283222, zero_point=0)
  (relu): Identity()
  (dequant): DeQuantize()
)
OrderedDict([('quant.scale', tensor([0.0430])), ('quant.zero_point', tensor([63])), ('conv.weight', tensor([[[[-0.2917,  0.0156,  0.2501],
          [ 0.1459,  0.1276,  0.3308],
          [ 0.0912,  0.2865,  0.1875]]]], size=(1, 1, 3, 3), dtype=torch.qint8,
       quantization_scheme=torch.per_channel_affine,
       scale=tensor([0.0026], dtype=torch.float64), zero_point=tensor([0]),
       axis=0)), ('conv.bias', Parameter containing:
tensor([-0.0878], requires_grad=True)), ('conv.scale', tensor(0.0084)), ('conv.zero_point', tensor(0))])
```

레이어의 종류가 바뀐것을 확인할 수 있음
zerop, scale이 추가됨
