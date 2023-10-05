import subprocess

import torch
import random
import numpy as np
from torch.ao.quantization import QConfig, HistogramObserver, PerChannelMinMaxObserver

from Torch_Circle_Mapper import Torch2CircleMapper
from Torch_Extractor import TorchExtractor


import urllib
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

from PIL import Image
from torchvision import transforms
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model


model_name = "mobilenet_v2"
model = torch.hub.load("pytorch/vision:v0.14.1", model_name,  pretrained=True)

input = input_batch
# input = torch.randn(1, 3, 224, 224)
# input = torch.randn(1, 3, 299, 299)

mapper = Torch2CircleMapper(original_model=model, sample_input=input, dir_path=model_name)
mapping, data = mapper.get_mapped_dict()

input_numpy = input.numpy()
with open(model_name + '/input0', 'wb') as fb:
    fb.write(bytes(input_numpy))

model.eval()
model.qconfig = QConfig(activation=torch.quantization.MinMaxObserver.with_args(dtype=torch.quint8),
                        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8))
p_model = torch.quantization.prepare(model)
p_model(input)
quant = torch.quantization.convert(p_model)
print(quant)

extractor = TorchExtractor(quant, json_path=model_name + '/qparam.json', qdtype=torch.quint8, partial_graph_data=data, mapping=mapping)
extractor.generate_files()