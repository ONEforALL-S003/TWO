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

# PyTorch Example Test Runner
import os
import torch
import importlib
from PTSQ import PTSQ
from pathlib import Path
import json
output_folder = "./output/"

Path(output_folder).mkdir(parents=True, exist_ok=True)
cases = {}
print(os.listdir('examples/'))
# example is operation name
for example in os.listdir('examples/'):
    # load example code
    module = importlib.import_module("examples." + example)
    
    model = PTSQ(module._model_)
    sample_input = module._dummy_
    
    model.eval()
    
    # now qconfig is fixed, but should be matched per operation
    backend = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    if backend == "qnnpack":
        torch.backends.quantized.engine = 'qnnpack'
    quantized_model = torch.quantization.prepare(model, inplace=False)

    try:
        quantized_model(sample_input)
        quantized_model = torch.quantization.convert(quantized_model)
        cases[example] = "Success"
    except Exception as ex:
        cases[example] = str(ex)
        print(ex)

with open("result.json","w") as f:
    json.dump(cases, f, indent=4)



