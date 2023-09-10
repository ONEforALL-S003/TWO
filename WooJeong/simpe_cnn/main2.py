import torch
import torch.nn as n
import torch.quantization
# torch.quantization is deprecated. Need to use torch.ao.quantization in latest torch, but we assume to use torch 1.7.0
from Net_Conv2d_2 import Net_Conv2d
import numpy
import onnx
import onnx_tf
import tensorflow as tf
import os

dir = "./out/"

if not os.path.exists("out"):
    os.mkdir(dir);

torch.manual_seed(123456)

input = torch.randn(4, 2, 4, 6)

model = Net_Conv2d()
torch.save(model, dir + "conv2d_original.pth")

onnx_path = dir + "conv2d_original.onnx"
torch.onnx.export(model, input, onnx_path, opset_version=9)
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)

inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
onnx.checker.check_model(inferred_model)
onnx.save(inferred_model, onnx_path)
tf_prep = onnx_tf.backend.prepare(inferred_model)
tf_prep.export_graph(path=dir + "conv2d_original.tf")
converter = tf.lite.TFLiteConverter.from_saved_model(dir + "conv2d_original.tf")
converter.allow_custom_ops = True
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()
open(dir + "conv2d_original.tflite", "wb").write(tflite_model)

model.eval()
state_dict = model.state_dict()
print(model)
tensor_name = state_dict.keys()
model.qconfig = torch.quantization.get_default_qconfig('x86')
p_model = torch.quantization.prepare(model)
p_model(input)
quantized = torch.quantization.convert(p_model)
quantized(input)
torch.save(quantized, dir + "conv2d_quantized.pth")

quant_state_dict = quantized.state_dict()
quant_value = {}
for name, tensor in quant_state_dict.items():
    if not tensor.is_quantized:
        continue
    data = {}
    if tensor.qscheme() == torch.per_tensor_affine:
        data['scale'] = tensor.q_scale()
        data['zop'] = tensor.q_zero_point()
    elif tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
        data['scale'] = tensor.q_per_channel_scales()
        data['zop'] = tensor.q_per_channel_zero_points()
        data['dim'] = tensor.q_per_channel_axis()

    if tensor.dtype == torch.qint8:
        data['value'] = torch.int_repr(tensor).numpy()
    else:
        data['value'] = tensor.numpy()
    quant_value[name] = data

print(quantized)
quantized_tensor_name = quant_state_dict.keys()
print(tensor_name)
print(quantized_tensor_name)