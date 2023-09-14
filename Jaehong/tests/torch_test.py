#!/usr/bin/env python

# Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# torch qparams extraction test

import torch
import torch.nn as nn
import onnx
import onnx_tf
import tensorflow as tf
import importlib
import argparse

from pathlib import Path

class QuantModel(nn.Module):
    def __init__(self, x):
        self.quant = torch.quantization.QuantStub()
        self.net = x
        self.dequant = torch.quantization.DeQuantStub()


    def forward(self, x):
        x = self.quant(x)
        x = self.net(x)
        x = self.dequant(x)
        return x
    

print("PyTorch version=", torch.__version__)
print("ONNX version=", onnx.__version__)
print("ONNX-TF version=", onnx_tf.__version__)
print("TF version=", tf.__version__)

parser = argparse.ArgumentParser(description='Process PyTorch python examples')

parser.add_argument('examples', metavar='EXAMPLES', nargs='+')

args = parser.parse_args()

output_folder = "./output/"

Path(output_folder).mkdir(parents=True, exist_ok=True)

for example in args.examples:
  print(example)