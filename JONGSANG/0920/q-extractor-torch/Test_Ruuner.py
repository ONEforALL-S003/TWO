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
import json
import os
import copy
import shutil

import torch
import numpy as np

import importlib
import argparse

from Torch_Circle_Mapper import Torch2CircleMapper
from Torch_Extractor import TorchExtractor

def check_np_shape(test:str, compare:str):
    test_np = np.load(test)
    compare_np = np.load(compare)

    if not test_np.shape == compare_np.shape:
        raise Exception

def check_operator(test:dict, compare:dict, test_path:str, compare_path:str):
    test_keys = test.keys()
    compare_keys = compare.keys()

    # if keys are different, extraction
    if (test_keys & compare_keys) != test_keys:
        raise Exception

    for key in test_keys:
        if key == 'weight':
            check_np_shape(os.path.join(test_path, test[key]), os.path.join(compare_path, compare[key]))
        elif key == 'zerop':
            check_np_shape(os.path.join(test_path, test[key]), os.path.join(compare_path, compare[key]))
        elif key == 'quantized_dimension':
            if test[key] != compare[key]:
                raise Exception
        elif key == 'value':
            check_np_shape(os.path.join(test_path, test[key]), os.path.join(compare_path, compare[key]))
        elif key == 'scale':
            check_np_shape(os.path.join(test_path, test[key]), os.path.join(compare_path, compare[key]))
        elif key == 'dtype':
            if test[key] != compare[key]:
                raise Exception
        else:
            raise Exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='q-extract-torch test runner')

    parser.add_argument('testcases', metavar='TESTCASES', nargs='+')
    args = parser.parse_args()

    test_work_dir = './test/tmp'
    test_compare_dir = './test/compare'

    if not os.path.exists(test_work_dir):
        os.mkdir(test_work_dir)

    dir_prefix = ['with_mapping', 'without_mapping']
    json_prefix = ['', 'not_mapped_']

    for testcase in args.testcases:
        module = importlib.import_module('test.example.' + testcase)
        model = module._model_
        original_model = copy.deepcopy(model)
        model.eval()
        model.qconfig = module._qconfig_
        p_model = torch.quantization.prepare(model)
        quantized_model = torch.quantization.convert(p_model)
        testcase_dir = os.path.join(test_work_dir, testcase)
        testcase_compare_dir = os.path.join(test_compare_dir, testcase)
        mapper = Torch2CircleMapper(
            original_model=original_model,
            sample_input=module._dummy_,
            dir_path=testcase_dir,
            tflite2circle_path='./tflite2circle',
            clean_circle=True)

        # {mapping, partial_graph_data}
        test_mapped_list = [
            mapper.get_mapped_dict(),
            [None, None]
        ]

        for i in range(2):
            current_test_path = testcase_dir + os.sep + dir_prefix[i]
            current_compare_path = testcase_compare_dir + os.sep + dir_prefix[i]
            json_paths = []
            for j in range(2):
                json_paths.append(current_test_path + os.sep + json_prefix[j] + 'qparm.json')
                json_paths.append(current_compare_path + os.sep + json_prefix[j] + 'qparm.json')

            extractor = TorchExtractor(
                quantized_model=quantized_model,
                json_path=json_paths[0],
                partial_graph_data=test_mapped_list[i][1])
            extractor.generate_files(test_mapped_list[i][0])

            flag = 0

            for j in range(2):
                try:
                    if os.path.exists(json_paths[2 * j]) ^ os.path.exists(json_paths[2 * j + 1]):
                        raise Exception
                    if not os.path.exists(json_paths[2 * j]):
                        continue
                    with open(json_paths[2 * j], 'r') as json_file:
                        test_json = json.load(json_file)
                    with open(json_paths[2 * j + 1], 'r') as json_file:
                        compare_json = json.load(json_file)
                    test_keys = test_json.keys()
                    compare_keys = compare_json.keys()

                    if (test_keys & compare_keys) != test_keys:
                        raise Exception

                    for key in test_keys:
                        check_operator(test_json[key], compare_json[key],
                                       current_test_path, current_compare_path)
                except:
                    if j == 0:
                        flag |= 0b01
                    elif j == 1:
                        flag |= 0b10
            if flag == 0b00:
                continue
            elif flag == 0b01:
                raise Exception('Test fails due to different mapping. (Maybe different onnx_tf version)')
            else:
                raise Exception('Test fails')

    shutil.rmtree(test_work_dir)


