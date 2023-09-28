import torch
import random
import numpy as np

from Torch_Circle_Mapper import Torch2CircleMapper
from Torch_Extractor import TorchExtractor

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)


model = torch.hub.load("pytorch/vision:v0.14.1", "mobilenet_v2",  pretrained=True)

input = torch.randn(1, 3, 244, 244)
print(model)

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