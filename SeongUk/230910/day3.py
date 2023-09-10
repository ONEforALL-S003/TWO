import torch
from torch import nn
from torch import quantization
from torch import onnx


'''
# 모델 생성
model = OneOperModel()

# 모델에 대한 입력값
x = torch.randn(1, 1, 224, 224, requires_grad=True)

# 모델 변환
torch.onnx.export(model,               # 실행될 모델
                  x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
                  "./SeongUk/230910/onnx_test.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
                  export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
                  opset_version=10,          # 모델을 변환할 때 사용할 ONNX 버전
                  do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
                  input_names = ['input'],   # 모델의 입력값을 가리키는 이름
                  output_names = ['output'], # 모델의 출력값을 가리키는 이름
                  dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
                                'output' : {0 : 'batch_size'}})


import onnx

onnx_model = onnx.load('./SeongUk/230910/onnx_test.onnx')
onnx.checker.check_model(onnx_model)

print(onnx_model)

'''


import torch
import onnx
import onnx_tf
import tensorflow as tf
import importlib
import argparse

from pathlib import Path

print("PyTorch version=", torch.__version__)
print("ONNX version=", onnx.__version__)
print("ONNX-TF version=", onnx_tf.__version__)
print("TF version=", tf.__version__)

output_folder = "./SeongUk/230910/output/"

Path(output_folder).mkdir(parents=True, exist_ok=True)

def convert(module, example):
    # save torch 모델
    torch.save(module, output_folder + example + ".pth")
    print("Generate '" + example + ".pth' - Done")

    opset_version = 9
    if hasattr(module, 'onnx_opset_version'):
        opset_version = module.onnx_opset_version()

    onnx_model_path = output_folder + example + ".onnx"


    # save onnx 모델
    torch.onnx.export(
        module, torch.randn(1, 2, 3, 3), onnx_model_path, opset_version=opset_version)
    print("Generate '" + example + ".onnx' - Done")

    onnx_model = onnx.load(onnx_model_path)
    onnx.checker.check_model(onnx_model)

    inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(inferred_model)
    onnx.save(inferred_model, onnx_model_path)


    # save tf 모델
    tf_prep = onnx_tf.backend.prepare(inferred_model)
    tf_prep.export_graph(path=output_folder + example + ".TF")
    print("Generate '" + example + " TF' - Done")


    # save tflite 모델
    converter = tf.lite.TFLiteConverter.from_saved_model(output_folder + example + ".TF")
    converter.allow_custom_ops = True
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

    tflite_model = converter.convert()
    open(output_folder + example + ".tflite", "wb").write(tflite_model)
    print("Generate '" + example + ".tflite' - Done")

example = 'OneOperModel'

# Operator가 하나인 모델 생성
class OneOperModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        return self.conv(x)
    
# load example code
module = OneOperModel()
convert(module, 'OneOperModel')

'''
# 양자화 된 모델 테스트
module.eval()
backend = "qnnpack"
module.qconfig = quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend
model_static_quantized = quantization.prepare(module, inplace=False)
model_static_quantized = quantization.convert(model_static_quantized, inplace=False)

convert(model_static_quantized, 'model_static_quantized')
'''


interpreter = tf.lite.Interpreter(model_path='./SeongUk/230910/output/OneOperModel.tflite')
import pprint
pprint.pprint(interpreter.get_tensor_details())