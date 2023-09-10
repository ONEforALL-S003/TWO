# PyTorch 모델 Quantization (토)

양자화된 모델에서는 scale, zero_point 등의 속성이 추가되어 있었다.

```
OneOperModel(
  (conv): QuantizedConv2d(1, 1, kernel_size=(1, 1), stride=(1, 1), scale=1.0, zero_point=0)
)
1.0
0
```

static양자화를 사용하였는데, torch문서상 conv2d는 dynamic 방식을 지원하지 않는다고 하여 static을 사용하였다.


[참고 문헌](https://tutorials.pytorch.kr/recipes/quantization.html)