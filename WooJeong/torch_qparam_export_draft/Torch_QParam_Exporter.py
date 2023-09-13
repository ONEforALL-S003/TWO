import os
import shutil
import sys

import torch
import torch.nn
import torch.quantization
import numpy as np
import collections
import json
import onnx
import onnx_tf
import tensorflow as tf
sys.path.append('./include')
import subprocess

#  generated by pics.
#  we need to set dependency on cmakelist
import flatbuffers
from include.circle.Model import Model

def permute(tensor):
    dim = len(tensor.shape)
    if dim == 4:  # NCHW to NHWC
        tensor = tensor.permute(0, 2, 3, 1)
    return tensor

class TorchQParamExporter:
    def __save_np(self, data):
        file_name = str(self.__np_idx) + ".npy"
        if data.shape==():
            data = np.array([data])
        np.save(os.path.join(self.__dir_path, file_name), data)
        self.__np_idx += 1
        return file_name

    def __init__(self, quantized_model, json_path):
        if quantized_model is None or not isinstance(quantized_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model")
        if json_path is None:
            raise Exception("Please specify save path")
        self.__json_path = json_path
        idx = json_path.rfind(os.path.sep)
        if idx == -1:
            self.__dir_path = ""
            self.__json_file_name = json_path
        else:
            self.__dir_path = json_path[:idx + 1]
            self.__json_file_name = json_path[idx + 1:]
        self.__np_idx = 0
        self.__data = {}
        self.__tree = collections.OrderedDict()
        self.__extract_module(module=quantized_model)
        self.__qdtype_mapping = {
            torch.quint8: {'str': "uint8", 'np': np.uint8},
            torch.qint8: {'str': "int8", 'np': np.int8},
            torch.qint32: {'str': "int32", 'np': np.int32}
        }
        self.__mapping = None
        self.__reverse_mapping = None

    def set_mapping(self, original_model, sample_input):
        if self.__mapping is not None or self.__reverse_mapping is not None:
            return

        self.__mapping = {}
        self.__reverse_mapping = reverse_mapping = {}

        if original_model is None or not isinstance(original_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model for mapping")
        if sample_input is None or not isinstance(sample_input, torch.Tensor):
            raise Exception("Please give sample input to convert model")
        params = original_model.named_parameters()

        for name, param in params:
            tensor = param.data
            tensor = permute(tensor)
            key = hash(tensor.numpy().tobytes())
            reverse_mapping[key] = name

        dir_path = os.path.join(self.__dir_path, "tmp")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        onnx_path = os.path.join(dir_path, "tmp.onnx")
        torch.onnx.export(original_model, sample_input, onnx_path, opset_version=9)
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.checker.check_model(inferred_model)
        onnx.save(inferred_model, onnx_path)
        tf_prep = onnx_tf.backend.prepare(inferred_model)
        tf_path = os.path.join(dir_path, 'tmp.tf')
        tf_prep.export_graph(path=tf_path)
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        tflite_model = converter.convert()
        tflite_path = os.path.join(dir_path, 'tmp.tflite')
        open(tflite_path, "wb").write(tflite_model)
        circle_path = os.path.join(dir_path, 'tmp.circle')
        try:
            subprocess.run(['./tflite2circle', tflite_path, circle_path], check=True)
        except:
            print('Fail to convert to circle')
        buf = bytearray(open(circle_path, 'rb').read())
        circle = Model.GetRootAsModel(buf)

        for idx in range(circle.SubgraphsLength()):
            self.__circle_subgraph_mapping_traverse(circle, circle.Subgraphs(idx))
        # shutil.rmtree(dir_path)

    def __circle_subgraph_mapping_traverse(self, circle, graph):
        mapping = self.__mapping
        reverse_mapping = self.__reverse_mapping

        for idx in range(graph.TensorsLength()):
            tensor = graph.Tensors(idx)
            name = tensor.Name().decode('utf-8')
            shape = tensor.ShapeAsNumpy()
            if shape.size == 0:
                continue
            buffer = circle.Buffers(tensor.Buffer()).DataAsNumpy()
            if type(buffer) is not np.ndarray or buffer.size == 0:
                continue
            key = hash(buffer.tobytes())

            if key in reverse_mapping:
                origin_name = reverse_mapping[key]
                mapping[origin_name] = name

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
            data[tensor_name] = permute(tensor)

    def save(self):
        tree = self.__tree
        mapped_data = {}
        not_mapped_data = {}
        if not os.path.exists(self.__dir_path):
            os.makedirs(self.__dir_path, exist_ok=True)

        mapping = self.__mapping

        for name, layer in tree.items():
            default_dtype = 'uint8'
            if "weight" in layer:
                w_name = name + '.weight'
                tensor = layer['weight']
                if w_name in mapping:
                    data = mapped_data
                    w_name = mapping[w_name]
                else:
                    data = not_mapped_data
                if tensor.is_quantized:
                    default_dtype = self.__qdtype_mapping[tensor.dtype]['str']
                    data[w_name] = self.__from_tensor(tensor=tensor)
            if "scale" in layer and "zero_point" in layer:
                # not sure about torch operator's scale and zero_point is for bias or input
                # let's assume operator's scale and zero_point is used for both bias and input
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
                    'dtype': default_dtype,
                    'quantized_dimension': 0
                }

                b_name = name + '.bias'
                if b_name in mapping:
                    b_name = mapping[b_name]
                    data = mapped_data
                else:
                    data = not_mapped_data

                if "bias" in layer:
                    quantized_bias = self.quantize_bias(layer['bias'], scale, zero_point)
                    data[b_name] = {
                        'scale': s_np,
                        'zerop': z_np,
                        'dtype': default_dtype,
                        'value': self.__save_np(quantized_bias),
                        'quantized_dimension': 0
                    }
        with open(self.__json_path, 'w') as json_file:
            json.dump(mapped_data, json_file)
        with open(os.path.join(self.__dir_path, 'not_mapped_' + self.__json_file_name), 'w') as json_file:
            json.dump(not_mapped_data, json_file)

    def __from_tensor(self, tensor):
        if tensor is None:
            raise Exception('tensor is null')
        data = {}
        if tensor.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            data['scale'] = self.__save_np(np.array(tensor.q_scale()))
            data['zerop'] = self.__save_np(np.array(tensor.q_zero_point()))
            data['quantized_dimension'] = 0
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