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
import tensorflow as tf


def quantize_numpy(tensor: np.ndarray, scale: np.ndarray, zero_point: np.ndarray,
                    dtype=np.int8):
    tensor = np.round(tensor / scale + zero_point)
    return tensor.astype(dtype)


def quantize_tensor(tensor: torch.Tensor, scale: np.ndarray, zero_point: np.ndarray,
                    dtype=np.int8) -> np.ndarray:
    if dtype not in (np.uint8, np.int8, np.int32):
        raise Exception('Please check dtype')
    return quantize_numpy(tensor.clone().detach().numpy(), scale, zero_point, dtype)


def quantize_tensor_per_channel(tensor: torch.Tensor, scale: np.ndarray, zero_point: np.ndarray,
                    dtype=np.int8) -> np.ndarray:
    # TODO: implement
    return tensor.numpy()


def to_numpy(tensor: torch.Tensor):
    if tensor.dtype == torch.qint8:
        return torch.int_repr(tensor).numpy()
    return tensor.numpy()


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
    def permute(buffer: numpy.ndarray, optype='') -> numpy.ndarray:
        rank = len(buffer.shape)
        # perm = list(range(2, rank)) + [1, 0]
        if rank == 4:
            if 'depthwise' in optype:  # NCHW to IHWC
                buffer = np.transpose(buffer, [1, 2, 3, 0])
            else:  # NCHW to NHWC
                buffer = np.transpose(buffer, [0, 2, 3, 1])
        return buffer

    @staticmethod
    def permute_dimension(rank, dimension, optype=''):
        if rank == 4:
            # https://github.com/onnx/onnx-tensorflow/blob/ee0c5e537b3cebbddc5773871e6786e6468c7f3f/onnx_tf/handlers/backend/conv_mixin.py#L101
            if 'depthwise' in optype:
                perm = [1, 2, 3, 0]
            else:
                perm = [0, 2, 3, 1]  # NCHW to NHWC
            return perm[dimension]
        return dimension

    @staticmethod
    def reshape(buffer, circle_shape):
        torch_shape = np.asarray(buffer.shape)
        if (torch_shape == circle_shape).all():
            return buffer

        torch_count = sum((torch_shape - 1) != 0)
        circle_count = sum((circle_shape - 1) != 0)

        if torch_count <= 1 and circle_count <= 1:
            buffer = buffer.reshape(circle_shape)
        else:
            raise Exception('shape not match')

        return buffer


    def __init__(self,
                 quantized_model: torch.nn.Module,
                 json_path: str,
                 mapping=None,
                 partial_graph_data=None):
        self.__np_idx = 0
        self.__input_dtype = None
        self.__graph_data = collections.OrderedDict()
        if mapping is None:
            self.__mapping = {}
        else:
            self.__mapping = mapping
        if partial_graph_data is None:
            self.__partial_graph_data = collections.OrderedDict()
        else:
            self.__partial_graph_data = partial_graph_data
        self.__json_path = json_path
        self.__dir_path, self.__json_file_name = os.path.split(json_path)
        self.__extract_module(quantized_model)

    def __extract_module(self, module: torch.nn.Module):
        graph_data = collections.OrderedDict()
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

        self.__preprocess(graph_data)
        return

    def __preprocess(self, graph_data=collections.OrderedDict):
        result = self.__graph_data = collections.OrderedDict()
        mapping = self.__mapping
        partial_data = self.__partial_graph_data

        if self.__input_dtype is None:
            raise Exception('Check network(torch model) have have tensor internally')


        q_dtype = self.qdtype_mapping[self.__input_dtype]
        q_dtype, dtype = q_dtype['np'], q_dtype['str']
        for name, layer in graph_data.items():
            if name in partial_data:
                circle = partial_data[name]
            else:
                circle = None

            # Batch Normalization
            if 'running_mean' in layer and 'running_var' in layer:
                if 'scale' not in layer or 'zero_point' not in layer:
                    continue

                scale = layer['scale'].numpy()
                zero_point = layer['zero_point'].numpy().astype(np.int64)

                """
                gamma -> tf's multiplier -> as 'weight' on PyTorch Batch Norm
                beta -> tf's offset -> as 'bias' on PyTorch Batch Norm

                tf -> tflite
                mul_float_data[i] = multiplier_float_data[i];
                add_float_data[i] = offset_float_data[i] - mean_float_data[i] * multiplier_float_data[i];
                """

                mul = layer['weight']
                add = layer['bias'] - layer['running_mean'] * mul
                add = quantize_tensor(add, scale, zero_point, q_dtype)

                add_name = name + '.bias'
                if add_name in mapping:
                    add_name = mapping[add_name]

                if circle is not None:
                    add_shape = circle['bias.shape']
                    mul_shape = circle['weight.shape']
                else:
                    add_shape = add.shape
                    mul_shape = mul.shape

                result[add_name] = {
                    'scale': scale,
                    'zerop': zero_point,
                    'quantized_dimension': 0,
                    'dtype': dtype,
                    'value': self.reshape(add, add_shape)
                }

                mul = quantize_tensor(mul, scale, zero_point, q_dtype)

                mul_name = name + '.weight'
                if mul_name in mapping:
                    mul_name = mapping[mul_name]

                result[mul_name] = {
                    'scale': scale,
                    'zerop': zero_point,
                    'quantized_dimension': 0,
                    'dtype': dtype,
                    'value': self.reshape(mul, mul_shape)
                }
                continue

            if 'weight' in layer:
                w_name = name + '.weight'
                tensor = layer['weight']
                if w_name in mapping:
                    w_name = mapping[w_name]

                if tensor.is_quantized:
                    result[w_name] = self.__process_weight(tensor, circle)

            if 'scale' in layer and 'zero_point' in layer:
                scale = layer['scale'].numpy()
                zero_point = layer['zero_point'].numpy().astype(np.int64)

                layer_name = name
                if layer_name in mapping:
                    layer_name = mapping[layer_name]

                result[layer_name] = {
                    'scale': scale,
                    'zerop': zero_point,
                    'dtype': dtype,
                    'quantized_dimension': 0
                }

                if 'bias' in layer and layer['bias'] is not None:
                    b_name = name + '.bias'
                    if b_name in mapping:
                        b_name = mapping[b_name]

                    bias = layer['bias']
                    bias = quantize_tensor(bias, scale, zero_point, np.int32)

                    if circle is not None:
                        bias_shape = circle['bias.shape']
                    else:
                        bias_shape = bias.shape

                    result[b_name] = {
                        'scale': scale,
                        'zerop': zero_point,
                        'dtype': 'int32',
                        'quantized_dimension': 0,
                        'value': self.reshape(bias, bias_shape)
                    }
            elif 'per_channel_scales' in layer and 'per_channel_zero_points' in layer and 'axis' in layer:
                scale = layer['per_channel_scales'].numpy()
                zero_point = layer['per_channel_zero_points'].numpy().astype(np.int64)

                layer_name = name
                if layer_name in mapping:
                    layer_name = mapping[layer_name]

                axis = layer['axis']
                if isinstance(axis, torch.Tensor):
                    axis = axis.numpy()[0]
                elif isinstance(axis, np.ndarray):
                    axis = axis[0]

                result[layer_name] = {
                    'scale': scale,
                    'zerop': zero_point,
                    'dtype': dtype,
                    'quantized_dimension': axis
                }

                if 'bias' in layer and layer['bias'] is not None:
                    b_name = name + '.bias'
                    if b_name in mapping:
                        b_name = mapping[b_name]

                    bias = layer['bias']
                    bias = quantize_tensor_per_channel(bias, scale, zero_point, axis, np.int32)
                    result[b_name] = {
                        'scale': scale,
                        'zerop': zero_point,
                        'dtype': 'int32',
                        'quantized_dimension': axis,
                        'value': bias
                    }

            # such as RELU or transpose like that, inherit quantization parameter
            elif 'prev_op' in layer:
                parent_name = graph_data[name]['prev_op']
                if parent_name not in result:
                    continue

                if parent_name + '.out' in mapping:
                    t_name = mapping[parent_name + '.out']
                else:
                    t_name = name

                result[t_name] = {
                    'scale': result[parent_name]['scale'],
                    'zerop': result[parent_name]['zerop'],
                    'dtype': result[parent_name]['dtype'],
                    'quantized_dimension': result[parent_name]['quantized_dimension']
                }

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

    # TODO: permute
    def __process_weight(self, tensor, circle=None):
        if tensor is None:
            raise Exception('tensor is null')
        data = {}

        optype = ''

        if circle is not None and 'optype' in circle:
            optype = circle['optype']

        if tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            data['scale'] = np.array(tensor.q_scale())
            data['zerop'] = np.array(tensor.q_zero_point()).astype(np.int64)
            data['quantized_dimension'] = 0
            data['value'] = self.permute(to_numpy(tensor), optype=optype)
        elif tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric,
                                  torch.per_channel_affine_float_qparams):
            data['scale'] = tensor.q_per_channel_scales().numpy()
            data['zerop'] = tensor.q_per_channel_zero_points().numpy().astype(np.int64)
            data['quantized_dimension'] = self.permute_dimension(rank=len(tensor.shape), dimension=tensor.q_per_channel_axis(),
                                                                 optype=optype)
            data['value'] = self.permute(to_numpy(tensor), optype=optype)

        data['dtype'] = self.qdtype_mapping[self.__input_dtype]['str']
        if circle is not None:
            circle_shape = circle['weight.shape']
            shape = data['value'].shape
            if not (shape == circle_shape).all():
                raise Exception("Different Shape")
        return data

    def generate_files(self):
        graph_data = self.__graph_data
        mapped_data = collections.OrderedDict()
        not_mapped_data = collections.OrderedDict()
        if not os.path.exists(self.__dir_path):
            os.makedirs(self.__dir_path, exist_ok=True)

        mapping = self.__mapping

        if 'NULL' in mapping:
            s_np = self.__save_np(np.array([0], dtype=np.float32))
            z_np = self.__save_np(np.array([0], dtype=np.int64))
            for null_op in mapping['NULL']:
                name = null_op[0]
                shape = null_op[1]
                mapped_data[name] = {
                    'scale': s_np,
                    'zerop': z_np,
                    'dtype': 'int32',
                    'quantized_dimension': 0,
                    'value': self.__save_np(np.zeros(shape=shape, dtype=np.int32))
                }

        for name, layer in graph_data.items():
            if name in mapping:  # torch name
                data = not_mapped_data
            else:  # circle name
                data = mapped_data
            element = data[name] = {}
            for attr_name, attr_value in layer.items():
                if isinstance(attr_value, np.ndarray):
                    element[attr_name] = self.__save_np(attr_value)
                else:
                    element[attr_name] = attr_value

        if len(mapped_data) > 0:
            with open(self.__json_path, 'w') as json_file:
                json.dump(mapped_data, json_file)
        if len(not_mapped_data) > 0:
            not_mapped_path = os.path.join(self.__dir_path,
                                           'not_mapped_' + self.__json_file_name)
            with open(not_mapped_path, 'w') as json_file:
                json.dump(not_mapped_data, json_file)