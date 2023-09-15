import torch
import torch.nn as n
import torch.quantization
# torch.quantization is deprecated. Need to use torch.ao.quantization in latest torch, but we assume to use torch 1.7.0
from Net_Conv2d import Net_Conv2d
import numpy
import onnx
import onnx_tf
import tensorflow as tf
import os
from Torch_QParam_Exporter import TorchQParamExporter

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
model.qconfig = torch.quantization.get_default_qconfig('x86')
p_model = torch.quantization.prepare(model)
p_model(input)
quantized = torch.quantization.convert(p_model)
quantized(input)
torch.save(quantized, dir + "conv2d_quantized.pth")

print(quantized)
exporter = TorchQParamExporter(quantized_model=quantized, json_path="export/Net_Conv2d/qparam.json")
