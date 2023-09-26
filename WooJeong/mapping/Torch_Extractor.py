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

import numpy
import torch
import torch.nn
import torch.quantization
import numpy as np
import collections
import json
import torch.nn.quantized.modules.linear


def quantize_tensor(tensor: torch.Tensor, scale, zero_point,
                    dtype=np.int8) -> torch.Tensor:
    if dtype not in (np.uint8, np.int8, np.int32):
        raise Exception('Please check dtype')
    new_tensor = tensor.clone().detach().numpy()
    new_tensor = new_tensor / scale + zero_point
    return new_tensor.astype(dtype)


class TorchExtractor:
    qdtype_mapping = {
        torch.quint8: {
            'str': "uint8",
            'np': np.uint8
        },
        torch.qint8: {
            'str': "int8",
            'np': np.int8
        },
        torch.qint32: {
            'str': "int32",
            'np': np.int32
        }
    }

    @staticmethod
    def permute(buffer: numpy.ndarray) -> numpy.ndarray:
        rank = len(buffer.shape)
        # perm = list(range(2, rank)) + [1, 0]
        if rank == 4:  # NCHW to NHWC
            buffer = np.transpose(buffer, [0, 2, 3, 1])
        return buffer

    @staticmethod
    def permute_dimension():
        print(1)

    @staticmethod
    def permute_dimension(rank, dimension):
        if rank == 4:
            perm = [0, 2, 3, 1]
            return perm[dimension]
        return dimension

    def __init__(self,
                 quantized_model: torch.nn.Module,
                 json_path: str,
                 partial_graph_data=None):
        self.__np_idx = 0
        self.__input_dtype = None
        self.__graph_data = collections.OrderedDict()
        if partial_graph_data is None:
            self.__partial_graph_data = collections.OrderedDict()
        else:
            self.__partial_graph_data = partial_graph_data
        self.__json_path = json_path
        self.__dir_path, self.__json_file_name = os.path.split(json_path)
        self.__extract_module(quantized_model)

    def __extract_module(self, module: torch.nn.Module):
        graph_data = self.__graph_data
        partial_graph_data = self.__partial_graph_data

        if len(module.state_dict()) == 0:
            raise Exception('There is no internal tensor in network')

        # Restructuring Neural Network model
        for name, mod in module.named_modules():
            # TODO: check whether there is better way to check instance of \
            #  torch.nn.quantized.modules.* and not torch.nn.modules.Module
            """
            Need to skip just Module. Only Operator/Tensor/Activation Needed
            When just using 'isinstance', all of operator/tensor/activation belong to it
            (All of them inherit torch.nn.modules.Module)

            Why '.nn.quantized.modules' instead of 'torch.nn.quantized.modules'?
            On previous version like 1.7.0, the path is 'torch.nn.quantized.modules',
            But on latest version, the path is 'torch.ao.nn.quantized.modules'
            """
            if name == '' or '.nn.quantized.modules' not in str(type(mod)):
                continue
            if isinstance(mod, torch.nn.quantized.modules.linear.LinearPackedParams):
                if self.__input_dtype is None:
                    self.__input_dtype = mod.dtype
                continue

            # get input's quantization type
            if self.__input_dtype is None and hasattr(mod, 'scale') and hasattr(
                    mod, 'zero_point') and hasattr(mod, 'dtype'):
                self.__input_dtype = mod.dtype

            if name in graph_data:
                data = graph_data[name]
            elif name in partial_graph_data:
                data = graph_data[name] = partial_graph_data[name]
            else:
                data = {}
                graph_data[name] = data
            for value_name, tensor in mod.state_dict().items():
                # Need to skip just Module. Only Operator/Tensor/Activation Needed
                # TODO: Find better way to check instance of torch.nn.quantized.modules
                if '.nn.quantized.modules' not in str(type(mod)):
                    continue
                tensor_name = value_name[value_name.rfind(".") + 1:]
                prefix = value_name[:value_name.rfind(".") + 1]
                # for Linear
                if prefix.find('_packed_params') != -1:
                    if tensor_name == '_packed_params':
                        data['weight'] = tensor[0]
                        data['bias'] = tensor[1]
                    continue

                # eg. bias None
                if tensor is None:
                    data[value_name] = None
                    continue

                if self.__input_dtype is None and tensor_name == 'weight':
                    self.__input_dtype = tensor.dtype

                data[tensor_name] = tensor

        # TODO: preprocess extracted data HERE!
        # such as permutation, dimension, reshape. Not generate file.
        print(graph_data)
        return

    def __save_np(self, data):
        file_name = str(self.__np_idx) + ".npy"
        if data.shape == ():
            data = np.array([data])
        if data.dtype == np.dtype(np.float64):
            data = data.astype(np.float32)
        np.save(os.path.join(self.__dir_path, file_name), data)
        self.__np_idx += 1
        return file_name

    def __from_tensor(self, tensor):
        if tensor is None:
            raise Exception('tensor is null')
        data = {}
        if tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            data['scale'] = self.__save_np(np.array(tensor.q_scale()))
            data['zerop'] = self.__save_np(np.array(tensor.q_zero_point()))
            data['quantized_dimension'] = 0
        elif tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric,
                                  torch.per_channel_affine_float_qparams):
            # TODO: permute
            data['scale'] = self.__save_np(tensor.q_per_channel_scales().numpy())
            data['zerop'] = self.__save_np(tensor.q_per_channel_zero_points().numpy())
            data['quantized_dimension'] = tensor.q_per_channel_axis()

        data['dtype'] = self.qdtype_mapping[tensor.dtype]['str']
        if tensor.dtype == torch.qint8:
            tensor = torch.int_repr(tensor)

        tensor_value = self.permute(tensor.numpy())
        data['value'] = self.__save_np(tensor_value)
        return data

    def generate_files(self, mapping=None):
        graph_data = self.__graph_data
        mapped_data = {}
        not_mapped_data = {}
        if not os.path.exists(self.__dir_path):
            os.makedirs(self.__dir_path, exist_ok=True)

        # method should work even there is no mapping data => all data will be not_mapped_data
        if mapping is None:
            mapping = {}

        if self.__input_dtype is None:
            raise Exception('Check network(torch model) have have tensor internally')

        if 'NULL' in mapping:
            z_np = self.__save_np(np.array([0], dtype=np.float32))
            for null_op in mapping['NULL']:
                name = null_op[0]
                shape = null_op[1]
                mapped_data[name] = {
                    'scale': z_np,
                    'zerop': z_np,
                    'dtype': 'int32',
                    'quantized_dimension': 0,
                    'value': self.__save_np(np.zeros(shape=shape, dtype=np.int32))
                }


        q_dtype= self.qdtype_mapping[self.__input_dtype]
        q_dtype, dtype = q_dtype['np'], q_dtype['str']
        for name, layer in graph_data.items():
            if 'running_mean' in layer and 'running_var' in layer:
                if 'scale' not in layer or 'zero_point' not in layer:
                    continue
                # TODO: permute / dimension

                scale = layer['scale'].numpy()
                s_np = self.__save_np(scale)
                zero_point = layer['zero_point'].numpy()
                z_np = self.__save_np(zero_point)

                """
                gamma -> tf's multiplier -> as 'weight' on PyTorch Batch Norm
                beta -> tf's offset -> as 'bias' on PyTorch Batch Norm
                
                tf -> tflite
                mul_float_data[i] = multiplier_float_data[i];
                add_float_data[i] = offset_float_data[i] - mean_float_data[i] * multiplier_float_data[i];
                """

                mul = layer['weight']
                add = layer['bias'] - layer['running_mean'] * mul
                add = quantize_tensor(add, scale=scale, zero_point=zero_point, dtype=q_dtype)

                add_name = name + '.bias'
                if add_name in mapping:
                    data = mapped_data
                    add_name = mapping[add_name]
                else:
                    data = not_mapped_data

                data[add_name] = {
                    'scale': s_np,
                    'zerop': z_np,
                    'quantized_dimension': 0,
                    'dtype': dtype,
                    'value': self.__save_np(add)
                }

                mul = quantize_tensor(mul, scale=scale, zero_point=zero_point, dtype=q_dtype)
                mul_name = name + '.weight'
                if mul_name in mapping:
                    data = mapped_data
                    mul_name = mapping[mul_name]
                else:
                    data = not_mapped_data

                data[mul_name] = {
                    'scale': s_np,
                    'zerop': z_np,
                    'quantized_dimension': 0,
                    'dtype': dtype,
                    'value': self.__save_np(mul)
                }

                continue

            if "weight" in layer:
                w_name = name + '.weight'
                tensor = layer['weight']
                if w_name in mapping:
                    data = mapped_data
                    w_name = mapping[w_name]
                else:
                    data = not_mapped_data
                if tensor.is_quantized:
                    data[w_name] = self.__from_tensor(tensor=tensor)
            if "scale" in layer and "zero_point" in layer:
                scale = layer['scale'].numpy()
                zero_point = layer['zero_point'].numpy()

                layer_name = name
                if layer_name in mapping:
                    layer_name = mapping[layer_name]
                    data = mapped_data
                else:
                    data = not_mapped_data

                s_np = self.__save_np(scale)
                z_np = self.__save_np(zero_point)
                data[layer_name] = {
                    'scale': s_np,
                    'zerop': z_np,
                    'dtype': dtype,
                    'quantized_dimension': 0
                }

                if layer['bias'] is not None:
                    b_name = name + '.bias'
                    if b_name in mapping:
                        b_name = mapping[b_name]
                        data = mapped_data
                    else:
                        data = not_mapped_data
                    bias = layer['bias']
                    bias = quantize_tensor(bias, scale, zero_point, dtype=np.int32)
                    data[b_name] = {
                        'scale': s_np,
                        'zerop': z_np,
                        'dtype': 'int32',
                        'quantized_dimension': 0
                    }
                    data[b_name]['value'] = self.__save_np(bias)
                # if bias is not None:
                #     bias = quantize_tensor(bias, scale, zero_point, dtype=np.int32)
                #     data[b_name]['value'] = self.__save_np(bias)

            # such as RELU or transpose like that, inherit quantization parameter
            elif 'prev_op' in layer:
                parent_name = graph_data[name]['prev_op']
                if parent_name in mapping and mapping[parent_name] in mapped_data:
                    parent = mapped_data[mapping[parent_name]]
                elif parent_name in not_mapped_data:
                    parent = not_mapped_data[parent_name]
                else:
                    continue

                if parent_name + '.out' in mapping:
                    t_name = mapping[parent_name + '.out']
                    data = mapped_data
                else:
                    t_name = name
                    data = not_mapped_data

                data[t_name] = {
                    'scale': parent['scale'],
                    'zerop': parent['zerop'],
                    'dtype': parent['dtype'],
                    'quantized_dimension': 0
                }
        if len(mapped_data) > 0:
            with open(self.__json_path, 'w') as json_file:
                json.dump(mapped_data, json_file)
        if len(not_mapped_data) > 0:
            not_mapped_path = os.path.join(self.__dir_path,
                                           'not_mapped_' + self.__json_file_name)
            with open(not_mapped_path, 'w') as json_file:
                json.dump(not_mapped_data, json_file)