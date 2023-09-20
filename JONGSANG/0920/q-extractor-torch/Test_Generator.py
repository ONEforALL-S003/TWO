# Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

import os
import copy

import torch

import importlib
import argparse

from Torch_Circle_Mapper import Torch2CircleMapper
from Torch_Extractor import TorchExtractor

# Test Data Generator
# Please check data manually by viewer(like netron)

parser = argparse.ArgumentParser(description='q-extract-torch test data generator')

parser.add_argument('testcases', metavar='TESTCASES', nargs='+')
args = parser.parse_args()

test_save_dir = './test/compare'

if not os.path.exists(test_save_dir):
    os.mkdir(test_save_dir)

test_mapping_dirs = ['with_mapping', 'without_mapping']

for testcase in args.testcases:
    module = importlib.import_module('test.example.' + testcase)
    model = module._model_
    original_model = copy.deepcopy(model)
    model.eval()
    model.qconfig = module._qconfig_
    p_model = torch.quantization.prepare(model)
    quantized_model = torch.quantization.convert(p_model)
    current_test_dir = os.path.join(test_save_dir, testcase)
    mapper = Torch2CircleMapper(
        original_model=original_model,
        sample_input=module._dummy_,
        dir_path=current_test_dir,
        tflite2circle_path='./tflite2circle',
        clean_circle=True)

    test_mapped_data = [
        # {mapping, partial_graph_data}
        mapper.get_mapped_dict(),
        [None, None]
    ]

    for i in range(2):
        json_path = current_test_dir + os.sep + test_mapping_dirs[i] + os.sep + 'qparm.json'
        extractor = TorchExtractor(
            quantized_model=quantized_model,
            json_path=json_path,
            partial_graph_data=test_mapped_data[i][1])
        extractor.generate_files(test_mapped_data[i][0])