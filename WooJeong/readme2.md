![torch model](https://github.com/ONEforALL-S003/TWO/assets/136890801/d402f55a-2d12-4ead-b61c-258473e94427)  
Quantization 이 이루어 지면 prefix로 Quantized가 붙으면서 Name이 바뀌는 것이라 생각하였으나, 실제로 확인해보았을 때 Quantization이 진행되도 Name은 그대로 있고 Layer(Operator) 종류가 바뀐다는 것을 확인할 수 있었음  
[Module](https://github.com/pytorch/pytorch/blob/89eb7a75a251c41c4bee86e9ede1001b0d3998af/torch/nn/modules/module.py#L2475)  
[Quantized Conv2d](https://github.com/pytorch/pytorch/blob/89eb7a75a251c41c4bee86e9ede1001b0d3998af/torch/ao/nn/quantized/modules/conv.py#L440) 
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/721f5ebf-f726-485c-a7f4-75269567698a)  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/3126e747-8cc7-4662-96b6-12696bdfb8b8)  
기존에 name이라 생각했던 건 단순히 어떤 layer/operator 인지를 나타내는 class 자체의 name 이었고, 실제 tensor/operator의 경로를 의미하는 module 내의 name은 변하지 않음  

```
odict_keys(['conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias', 'conv3.weight', 'conv3.bias'])
odict_keys(['quant.scale', 'quant.zero_point', 'conv1.weight', 'conv1.bias', 'conv1.scale', 'conv1.zero_point', 'conv2.weight', 'conv2.bias', 'conv2.scale', 'conv2.zero_point', 'conv3.weight', 'conv3.bias', 'conv3.scale', 'conv3.zero_point'])
```
quantization을 통하여 Layer에 변동이 생기더라도, state_dict.keys() 를 이용하여 tensor의 이름들을 확인하면 quantization parameter와 연관 된 tensor가 추가되고, 기존에 존재하던 tensor는 그래도 있는 것을 확인 할 수 있음  

![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/e7b39961-4842-4733-b282-326b7e8b0e21)
또한 이유는 모르겠지만 tensor의 정보를 확인하였을 때 명확하게 dtype이 qint8로 잡혀있음에도 실제 tensor data 값을 확인하면 float 형식으로 나오는 것을 확인할 수 있는데, reference를 찾아보고 torch code를 찾아 봐도 정확한 이유는 알 수 없음.  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/4ff2088c-8463-43fe-8973-0b3bc97a3421)
Quantization 이전의 weight 값과 비교해보았을 때, 어느 정도 유사한 것을 보아 해당 값은 실제 quantization 된 값을 보여주는 것이 아니라 사용자 편의를 위해 quantization 값을 dequantize 하여 보여주는 것임을 유추할 수 있음.  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/5ee1d96b-fce0-4404-9b6e-3eb260434a2c)  
[Tensor Storage](https://github.com/pytorch/pytorch/blob/89eb7a75a251c41c4bee86e9ede1001b0d3998af/torch/storage.py#L367C39-L367C39)  
실제 Tensor 내의 Data는 Quantization 된 qint8 값이 제대로 들어가는 있는 것을 보아, print에 찍히는 값은 편의를 위한 dequantize 값이라는 강한 의심이 됨  

Tensor 값을 접근하려면 Tensor가 우선 CPU에 있어야 하며(GPU에 있는 건 detach 하여 cpu로 옮겨 줘야한다고 함) .numpy() 를 호출하여 numpy로 변환하여 사용해야하나, qint8만 그런 것인지는 모르겠지만(Reference를 추가적으로 찾아 봤을 때 qint8만 안 되는 것일 가능성이 매우 높음)  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/91a8ec84-4315-43df-9019-80b9c4f03667)  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/5ac71c64-1f17-4620-b75d-5e1fd653467b)  
qint8의 경우 numpy로 변환이 그냥은 안 되는 것을 확인할 수 있음  

[torch.int_repr](https://pytorch.org/docs/stable/generated/torch.Tensor.int_repr.html#torch.Tensor.int_repr)  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/a88968c9-a4d9-4ec5-aee4-72401057fe67)  
torch.int_repr 을 사용하여 qint8 형식의 tensor를 실제 int8 형식으로 변환할 수 있음(Reference 상에 qint8에 관한 부분만 있어서, qint16을 비롯해 다른 type은 tensor 자체가 소수로 안 나오고 정상적일 것이라 추정됨...)  
int_repr를 사용하면서 is_quantized 속성도 풀리므로, 최종적으로 실제 value를 뽑을 때 변환하여 사용해야하는 것을 알 수 있음  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/16c6f41c-4115-4f27-8d15-e42c7793f9a0)
결론적으로 int_repr에 numpy() 붙여주면 qint8 형식을 numpy로 변환할 수 있음  


![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/5bc5bf9d-181c-4a8d-a160-b0b774a4a2d9)  
추가적으로 Tensorflow에서는 Bias 도 Quantization이 이루어지는 것으로 확인 했었는데, torch에서는 공식 문서를 확인해도 Bias가 Quantization 이루어진 다는 것은 확인할 수 없었음  
[Quantization Mode Support](https://pytorch.org/docs/stable/quantization.html#quantization-mode-support)  
[aten/src/Aten/native/quantized/cpu/qconv.cpp](https://github.com/pytorch/pytorch/blob/89eb7a75a251c41c4bee86e9ede1001b0d3998af/aten/src/ATen/native/quantized/cpu/qconv.cpp#L698C83-L698C96)  
Bias 자체는 Torch Backend 단에서 Inference 때 quantize 한다는 것 같은데, circle은 backend(runtime) 때 quantize 하는 것이 아니라 .circle 내에 quantized bias를 가지고 있어야함으로 미리 계산해서 넣어줄 필요성이 생각되어 짐  
하지만 정확하게 bias를 위한 quantization parameter를 정확하게 reference/documentation 상으론 찾지는 못 함 (찾아야함)  

![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/5646ecca-a463-402b-82f1-98947128940d)
weight tensor 내에 scale과 zero point가 존재하고, Tensor내의 scale, zero point, weight value를 이용하여 계산해본 dequantized 값과 original value가 유사한 것을 보았을 때 각 Tensor 내의 scale, zero point를 사용하는 것은 우선 맞아 보임  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/5c69dd5b-a7fa-450b-b0b7-02741ed944a2)  
[![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/a892dab4-d266-4524-b3be-8e1c4e3909d5)](https://www.wolframalpha.com/input?i=y+%3D+x+%2Fs+%2B+z%2C+y+%3D+-72%2C+s+%3D+0.0027089817449450493%2C++z+%3D+0%2C++x+%3D+%3F)  

Bias는 quantized 가 아니기 때문에 scale과 zero point를 가지고 있지 않음  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/3eb77eeb-1483-4dfc-82ce-268f763a47f0)  
Netron으로 한번 확인해보려고 해도 attribute가 int 값으로 나와 정확하게 확인할 수 없으며, onnx 모델로 변환해보려고 해도 quantized model은 onnx로 변환이 안 되는 것 같음.  
아마 모델을 전체 양자화를 걸어서 conv2d에 input으로 들어오는 게 qint 형식이여서 .scale .zero_point가 bias를 위한 quantization parameter라고 생각은 되나 조금 더 확인이 필요할 것 같아 보임  
Dynamic Quantization은 layer 가 얇으면 quantization이 잘 안 되는 것 같아서 static으로 했는데, 조금 더 깊게 or conv2d 말고 dense를 이용해서 quant stub 안 주고 했을 때 어떻게 되는지 확인할 필요성이 있어 보임  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/8db8b7f8-4841-4d2c-81fd-b884081f11f5)  
![image](https://github.com/ONEforALL-S003/TWO/assets/136890801/4ed0b674-54fb-4f17-a3d2-9fa0621a1304)  
Static Quantization 때문에 전체 양자화 되서 operator에 들어오는 input을 inference에 quantization 해줄 필요 없으니 operator에 scale 과 zero point가 없으면, operator는 input tensor를 위한 거고 state_dict로 조회한 tensor에 있는 건 bias를 위한거다라고 생각해볼 수도 있을 텐데  
그렇다기엔 또 동일한 값을 가지고 있어서 조금 더 살펴볼 필요성이 있어 보임...  
