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

import collections
import copy
import subprocess
import sys
import os
import inspect

import numpy as np
import torch
import torch.nn
import torch.quantization
from torch._C._onnx import TrainingMode


#  generated by pics.
#  TODO: we need to set pics dependency on cmakelist
sys.path.append('./include')
from include.circle.Model import Model
from include.circle.SubGraph import SubGraph
from include.circle.TensorType import TensorType
from include.circle.BuiltinOperator import BuiltinOperator


class Torch2CircleMapper:
    def __init__(self,
                 original_model: torch.nn.Module,
                 sample_input: torch.Tensor,
                 dir_path: str):
        self.__dir_path = dir_path

        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        self.__mapping = None
        self.__reverse_mapping = None
        self.__network_input = None
        self.__network_output = None

        if original_model is None or not isinstance(original_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model for mapping")

        self.__original_model = original_model
        self.__sample_input = sample_input
        self.__partial_graph_data = collections.OrderedDict()

    def get_mapped_dict(self):
        if self.__mapping is not None:
            return self.__mapping, self.__partial_graph_data
        original_model = self.__original_model
        sample_input = self.__sample_input
        dir_path = self.__dir_path
        onnx_path = os.path.join(dir_path, 'tmp.onnx')
        circle_path = os.path.join(dir_path, 'input.circle')

        original_model = copy.deepcopy(original_model)
        self.__original_model = original_model
        device = torch.device('cpu')
        original_model = original_model.to_empty(device=device)
        original_model._apply(lambda t: t.detach_())

        self.__mapping = {}
        reverse_mapping = self.__reverse_mapping = {}

        visit = set()

        idx = 100

        for name, mod in original_model.named_modules():
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html#batchnorm2d
            if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
                # gamma -> tf's multiplier -> as 'weight' on PyTorch Batch Norm
                # beta -> tf's offset -> as 'bias' on PyTorch Batch Norm
                mod.affine = False
                mod.training = False
                mod.num_batches_tracked = None
                mod.track_running_stats = False
                visit.add(name + '.running_mean')
                mod.running_mean.numpy().fill(0)
                visit.add(name + '.running_var')
                mod.running_var.numpy().fill(1)
                mod.eps = 0
                mod.momentum = 0  # 1 - momentum -> should be 0 to be converted as 1
                mod.weight.numpy().fill(idx)
                visit.add(name + '.weight')
                reverse_mapping[idx] = name + '.mul.weight'
                idx = idx + 1
                mod.bias.numpy().fill(idx)
                visit.add(name + '.bias')
                reverse_mapping[idx] = name + '.add.bias'
                idx = idx + 1

        for name, tensor in original_model.state_dict().items():
            if name in visit:
                continue
            visit.add(name)
            buffer = tensor.numpy()
            buffer.fill(idx)
            reverse_mapping[idx] = name
            idx = idx + 1

        torch.save(original_model.state_dict(), os.path.join(dir_path, 'tmp.pth'))
        onnx_saved = False
        for onnx_opset in range(9, 16):
            try:
                torch.onnx.export(
                    original_model,
                    sample_input,
                    onnx_path,
                    training=TrainingMode.PRESERVE,  # torch/onnx/utils 1164
                    export_params=True,
                    opset_version=onnx_opset,
                    do_constant_folding=False
                )
                onnx_saved = True
                break
            except Exception as ex:
                print(ex)

        if not onnx_saved:
            raise Exception

        try:
            subprocess.run(['python3', 'one-import-onnx.py', '-i', onnx_path, '-o', circle_path])
        except Exception as ex:
            print(ex)
            raise Exception

        buf = bytearray(open(circle_path, 'rb').read())
        circle = Model.GetRootAsModel(buf)
        self.__generate_mapped_dict(circle)
        return self.__mapping, self.__partial_graph_data

    def __generate_mapped_dict(self, circle):
        original_model = self.__original_model

        self.__network_input = []
        self.__network_output = []
        for idx in range(circle.SubgraphsLength()):
            self.__circle_subgraph_mapping_traverse(circle, circle.Subgraphs(idx))

        graph_data = self.__partial_graph_data

        input_list = []
        output_list = []
        prev_module_name = None
        tmp_names = []
        for name, mod in original_model.named_modules():
            # it's just model itself
            if name == '':
                continue
            class_name = str(type(mod))
            if '.nn.modules.container' in class_name or '.nn.modules.module' in class_name:
                continue
            tmp_names.append(name)
            if isinstance(mod, torch.quantization.QuantStub):
                input_list.append(name)
            elif isinstance(mod, torch.quantization.DeQuantStub):
                output_list.append(name)
            # Operator which don't have tensors
            elif len(mod.state_dict()) == 0:
                # activation such as RELU, don't have tensor. So it can't be mapped
                # use previous operator data to map it
                if name not in graph_data:
                    graph_data[name] = {}
                graph_data[name]['prev_op'] = prev_module_name
            prev_module_name = name

        circle_input = self.__network_input
        if len(input_list) == 1 and len(circle_input) == 1:
            self.__mapping[input_list[0]] = circle_input.Name().decode('utf-8')
        # Even there is no QuantStub mapping works
        elif len(input_list) == 0 and len(circle_input) == 1:
            self.__partial_graph_data['input'] = circle_input[0].Name().decode('utf-8')
        else:
            print("There are more than one input in Network. Please map it manually")

        circle_output = self.__network_output
        if len(output_list) == 1 and len(circle_output) == 1:
            self.__mapping[output_list[0]] = circle_output.Name().decode('utf-8')
        # Even there is no DeQuantStub mapping works
        elif len(output_list) == 0 and len(circle_input) == 1:
            self.__partial_graph_data['output'] = circle_output[0].Name().decode('utf-8')
        else:
            print("There are more than one output in Network. Please map it manually")

    def __circle_subgraph_mapping_traverse(self, circle: Model, graph: SubGraph):
        mapping, reverse_mapping = self.__mapping, self.__reverse_mapping
        # For operators those not have value
        op_mapping = {}
        graph_data = self.__partial_graph_data

        # get input tensors of graph
        for idx in range(graph.InputsLength()):
            input_tensor = graph.Tensors(graph.Inputs(idx))
            self.__network_input.append(input_tensor)

        # get output tensors of graph
        for idx in range(graph.OutputsLength()):
            output_tensor = graph.Tensors(graph.Outputs(idx))
            self.__network_output.append(output_tensor)

        dtype_resolver = {}

        for i in inspect.getmembers(TensorType):
            if not i[0].startswith('_') and not inspect.ismethod(i[1]):
                dtype_resolver[i[1]] = i[0].lower()

        builtin_op_resolver = {}
        for i in inspect.getmembers(BuiltinOperator):
            if not i[0].startswith('_') and not inspect.ismethod(i[1]):
                builtin_op_resolver[i[1]] = i[0].lower()
        # get all tensors from graph
        for idx in range(graph.TensorsLength()):
            tensor = graph.Tensors(idx)
            tensor_dtype = dtype_resolver[tensor.Type()]
            np_dtype = np.dtype(tensor_dtype)
            name = tensor.Name().decode('utf-8')
            shape = tensor.ShapeAsNumpy()
            # When the tensor don't have shape, We can't map it due to lack of tensor value
            if shape.size == 0:
                continue
            buffer = circle.Buffers(tensor.Buffer()).DataAsNumpy()
            # When fetched buffer is not type of numpy or size is 0 -> The tensor actually have no value
            if type(buffer) is not np.ndarray or buffer.size == 0:
                continue
            buffer = np.frombuffer(buffer, dtype=np_dtype)
            key = buffer[0]

            # If not, it does not belong to our marked PyTorch's Tensor
            if not np.all(buffer == key):
                continue

            key = round(key)

            if key == 0:
                if 'NULL' not in graph_data:
                    graph_data['NULL'] = []
                graph_data['NULL'].append([name, shape])
                continue

            # If equivalent torch tensor of current circle tensor, we can map it
            if key in reverse_mapping:
                origin_name = reverse_mapping[key]  # torch's name
                if origin_name not in mapping:
                    mapping[origin_name] = []
                mapping[origin_name].append(name)  # mapping torch name to circle tensor name
                origin_operation_name = origin_name[:origin_name.rfind(".")]
                origin_tensor_name = origin_name[origin_name.rfind(".") + 1:]
                if origin_operation_name not in graph_data:
                    graph_data[origin_operation_name] = {}
                graph_data[origin_operation_name][origin_tensor_name + '.shape'] = shape

                # To map tensor's those whom don't have tensor value, memorize tensor data(buffer index)
                if origin_operation_name not in op_mapping:
                    op_mapping[origin_operation_name] = set()
                op_mapping[origin_operation_name].add(idx)

        # approximately it takes O(N^2)
        # we need to think to it better way or not
        # TODO: maybe Trie will works. Check it whether it works or not
        for i in range(graph.OperatorsLength()):
            operator = graph.Operators(i)
            opcode = operator.OpcodeIndex()
            opcode = circle.OperatorCodes(opcode).BuiltinCode()
            builtin_op_name = builtin_op_resolver[opcode]

            # get operator's input tensor's indexes
            input_set = set(operator.InputsAsNumpy().tolist())

            for op_name, op_input in op_mapping.items():
                # When there is subset of already mapped tensor's indexes
                # That mapped subset operator information is same with current operation
                # Then we can map torch operator name to circle's operator name
                if input_set.issuperset(op_input):
                    input_set = input_set - op_input

                    for tensor_idx in input_set:
                        tensor = graph.Tensors(tensor_idx)
                        tensor_name = tensor.Name().decode('utf-8')
                        # torch operator name -> circle operator name
                        if op_name not in mapping:
                            mapping[op_name] = []
                        mapping[op_name].append(tensor_name)
                        if op_name not in graph_data:
                            graph_data[op_name] = {}
                        graph_data[op_name]['optype'] = builtin_op_name


                    if op_name + '.out' not in mapping:
                        mapping[op_name + '.out'] = []

                    for output_idx in range(operator.OutputsLength()):
                        output_tensor = graph.Tensors(operator.Outputs(output_idx))
                        output_tensor_name = output_tensor.Name().decode('utf-8')
                        mapping[op_name + '.out'].append(output_tensor_name)
                    break
