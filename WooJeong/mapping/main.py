import subprocess

import torch
import random
import numpy as np

from Torch_Circle_Mapper import Torch2CircleMapper
from Torch_Extractor import TorchExtractor

from torch._C._onnx import TrainingMode

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)


model = torch.hub.load("pytorch/vision:v0.14.1", "mobilenet_v2",  pretrained=True)

input = torch.randn(1, 3, 244, 244)

# for onnx_opset in range(9, 16):
#     try:
#         torch.onnx.export(
#             model,
#             input,
#             'preserve/model.onnx',
#             training=TrainingMode.PRESERVE,  # torch/onnx/utils 1164
#             export_params=True,
#             opset_version=onnx_opset,
#             do_constant_folding=False
#         )
#         onnx_saved = True
#         break
#     except Exception as ex:
#         print(ex)
#
# if not onnx_saved:
#     raise Exception
#
# subprocess.run(['python3', 'one-import-onnx.py', '-i', 'preserve/model.onnx', '-o', 'preserve/tmp.circle'])

mapper = Torch2CircleMapper(original_model=model, sample_input=input, dir_path='preserve')
mapping, data = mapper.get_mapped_dict()

model.eval()
model.qconfig = torch.quantization.get_default_qconfig('x86')
p_model = torch.quantization.prepare(model)
p_model(input)
quant = torch.quantization.convert(p_model)

extractor = TorchExtractor(quant, json_path='preserve/qparam.json', partial_graph_data=data, mapping=mapping)
extractor.generate_files()
print(1)