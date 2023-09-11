import os
import torch
import torch.nn
import torch.quantization
import numpy as np
import collections
import json


def permute(tensor):
    dim = len(tensor.shape)
    if dim == 4:  # NCHW to NHWC
        tensor = tensor.permute(0, 3, 1, 2)
    return tensor


class TorchQParamExporter:
    def __save_np(self, data):
        file_name = str(self.__np_idx) + ".npy"
        np.save(os.path.join(self.__dir_path, file_name), data)
        self.__np_idx += 1
        return file_name

    def __init__(self, quantized_model, json_path, default_dtype=torch.quint8):
        if quantized_model is None or not isinstance(quantized_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model")
        if json_path is None:
            raise Exception("Please specify save path")
        self.__default_dtype = default_dtype
        self.__json_path = json_path
        idx = json_path.rfind(os.path.sep)
        if idx == -1:
            self.__dir_path = ""
        else:
            self.__dir_path = json_path[:idx + 1]
        self.__np_idx = 0
        self.__data = {}
        self.__tree = collections.OrderedDict()
        self.__extract_module(module=quantized_model)
        self.__qdtype_mapping = {
            torch.quint8: {'str': "uint8", 'np': np.uint8},
            torch.qint8: {'str': "int8", 'np': np.int8},
            torch.qint32: {'str': "int32", 'np': np.int32}
        }
        self.save()

    def __extract_module(self, module):
        tree = self.__tree
        for name, tensor in module.state_dict().items():
            layer = name[:name.rfind(".")]
            if layer in tree:
                data = tree[layer]
            else:
                data = {}
                tree[layer] = data
            tensor_name = name[name.rfind(".") + 1:]
            data[tensor_name] = tensor

    def save(self):
        tree = self.__tree
        data = {}
        if not os.path.exists(self.__dir_path):
            os.makedirs(self.__dir_path, exist_ok=True)

        for name, layer in tree.items():
            if "weight" in layer:
                tensor = permute(layer['weight'])
                if tensor.is_quantized:
                    data[name + ".weight"] = self.__from_tensor(tensor=tensor)
            if "scale" in layer and "zero_point" in layer:
                # not sure about torch operator's scale and zero_point is for bias or input
                # let's assume operator's scale and zero_point is used for both bias and input
                scale = permute(layer['scale']).numpy()
                zero_point = permute(layer['zero_point']).numpy()
                data[name] = {
                    'scale': self.__save_np(scale),
                    'zerop': self.__save_np(zero_point),
                    'dtype': self.__qdtype_mapping[self.__default_dtype]['str']
                }

                if "bias" in layer:
                    quantized_bias = self.quantize_bias(permute(layer['bias']), scale, zero_point)
                    data[name + '.bias'] = data[name].copy()
                    data[name + '.bias']['value'] = self.__save_np(quantized_bias)
        with open(self.__json_path, 'w') as json_file:
            json.dump(data, json_file)

    def __from_tensor(self, tensor):
        if tensor is None:
            raise Exception('tensor is null')
        data = {}
        if tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            data['scale'] = self.__save_np(np.array(tensor.q_scale()))
            data['zerop'] = self.__save_np(np.array(tensor.q_zero_point()))
        elif tensor.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric, torch.per_channel_affine_float_qparams):
            data['scale'] = self.__save_np(tensor.q_per_channel_scales().numpy())
            data['zerop'] = self.__save_np(tensor.q_per_channel_zero_points().numpy())
            data['quantized_dimension'] = tensor.q_per_channel_axis()

        # https://pytorch.org/docs/stable/quantization.html#quantized-tensor
        # https://pytorch.org/docs/1.7.0/quantization.html#quantized-tensors
        # According to documentation, pytorch latest tensor support quint8, qint8, qint32, float16
        # But we use pytorch 1.7.0 due to onnx-tf, pytorch 1.7.0 do not support float16
        if tensor.dtype == torch.qint8:
            data['value'] = self.__save_np(torch.int_repr(tensor).numpy())
        else:
            data['value'] = self.__save_np(tensor.numpy())
        data['dtype'] = self.__qdtype_mapping[tensor.dtype]['str']
        return data

    def quantize_bias(self, tensor, scale, zero_point, dtype=np.int8):
        if dtype not in (np.uint8, np.int8, np.int32):
            raise Exception('Check dtype of bias quantization')
        bias = tensor.clone().detach().numpy()
        bias = bias / scale + zero_point
        return bias

