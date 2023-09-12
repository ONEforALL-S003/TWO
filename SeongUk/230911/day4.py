import torch
from torch import nn
from torch import quantization


# Operator가 하나인 모델 생성
class OneOperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(1, 1, 1)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.dequant(x)
        return x

# 모델 생성
model = OneOperModel()

model.eval()

backend = "qnnpack"
model.qconfig = quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

model_static_quantized = quantization.prepare(model, inplace=False)
model_static_quantized = quantization.convert(model_static_quantized, inplace=False)


import pprint
# 모델 출력
print('OneOperModel_quant')
print(model_static_quantized)

temp_dict = {}

for key, module in model_static_quantized._modules.items():
    if key == 'conv':
        w_b = module._weight_bias()
        pprint.pprint(w_b)
        weight = w_b[0]
        data = weight.data
        print(type(w_b))
        print(type(weight))
        print(type(w_b[1]))
        
        
        if weight.qscheme() == torch.per_tensor_affine:
            quantizer_params = weight.qscheme(), \
                weight.q_scale(), \
                weight.q_zero_point()
        elif weight.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
            quantizer_params = weight.qscheme(), \
                weight.q_per_channel_scales(), \
                weight.q_per_channel_zero_points(), \
                weight.q_per_channel_axis()
        else:
            raise RuntimeError(f"Unsupported qscheme {weight.qscheme()} in deepcopy")

        print(quantizer_params)        

        data1 = {
            "dtype": weight.dtype,
            "scale": "0.npy", # dtype: fp32, shape: [1]
            "zerop": "1.npy", # dtype: int64, shape: [1]
            "quantized_dimension": 0
        }
        temp_dict['convolution;PartitionedCall/convolution'] = data1