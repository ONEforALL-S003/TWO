import sys

import numpy

sys.path.append("./include")

# consider whether flatbuffers can be included as python package
# or import flatbuffers as module from one/res/externals/FLATBUFFERS
# maybe we need to include as module from external due to flac
# (maybe we should to build tflite schema at nncc configure stage,
#  and both version of schema_generated and flatbuffers must be same)
import flatbuffers as fb
from include.tflite.Model import Model as tfModel

import os
import numpy as np
import json


class TfFbParamExporter:
    # Need to update dictionary when Tensorflow Lite Tensor Type Updated
    tensor_resolver = {
        0: ['FLOAT32', np.float32],
        1: ['FLOAT16', np.float16],
        2: ['INT32', np.int32],
        3: ['UINT8', np.uint8],
        4: ['INT64', np.int64],
        5: ['STRING', np.string_],
        6: ['BOOL', np.bool_],
        7: ['INT16', np.uint16],
        8: ['COMPLEX64', np.complex64],
        9: ['INT8', np.int8],
        10: ['FLOAT64', np.float64],
        11: ['COMPLEX128', np.complex128],
        12: ['UINT64', np.uint64],
        13: ['RESOURCE', ],  # TODO Not sure about what to give
        14: ['VARIANT', ],  # TODO Not sure about what to give
        15: ['UINT32', np.uint32]
    }

    def __save_np(self, data):
        file_name = str(self.__np_idx) + ".npy"
        np.save(os.path.join(self.__dir_path, file_name), data)
        self.__np_idx += 1
        return file_name

    def __init__(self, json_path, model_path=None, model_content=None):
        self.__json_path = json_path
        idx = json_path.rfind(os.path.sep)
        if idx == -1:
            self.__dir_path = ""
        else:
            self.__dir_path = json_path[:idx + 1]
        self.__np_idx = 0
        if model_path is None and model_content is None:
            raise Exception('There is no model')
        elif model_content is not None:
            buf = model_content
        else:  # model_path
            buf = open(model_path, 'rb').read()
            buf = bytearray(buf)
        self.__model = tfModel.GetRootAsModel(buf, 0)
        self.__subGraph = []
        for idx in range(self.__model.SubgraphsLength()):
            tensors = self.__sub_graph_traverse(self.__model.Subgraphs(idx))
            self.__subGraph.append(tensors)

    def __sub_graph_traverse(self, graph):
        tensors = []
        for idx in range(graph.TensorsLength()):
            tensor = graph.Tensors(idx)
            tensor_type = TfFbParamExporter.tensor_resolver[tensor.Type()]
            shape = tensor.ShapeAsNumpy()
            name = tensor.Name()
            buffer = self.__model.Buffers(tensor.Buffer())
            quantization = tensor.Quantization()
            if quantization is None:
                continue
            scale_legnth = quantization.ScaleLength()
            zerop_length = quantization.ZeroPointLength()
            if scale_legnth == 0 and zerop_length == 0:  # not quantized
                continue
            scales = quantization.ScaleAsNumpy()
            zerop = quantization.ZeroPointAsNumpy()
            if scales.size != scale_legnth or zerop.size != zerop_length:
                raise Exception('Quantization Conflict')
            dimension = quantization.QuantizedDimension()
            data = {
                'dtype': tensor_type[0],
                'name': str(name, 'utf-8'),  # TODO Wee need to consider whether trim prefix/suffix or not
                'scale': scales,
                'zerop': zerop,
                'quantized_dimension': dimension,
                'value': 0
            }
            value = buffer.DataAsNumpy()
            if type(value) == numpy.ndarray:
                value = np.frombuffer(value, dtype=tensor_type[1])
                value = np.reshape(value, shape)
                data['value'] = value
            tensors.append(data)
        return tensors

    def save(self):
        data = {}
        if not os.path.exists(self.__dir_path):
            os.makedirs(self.__dir_path)
        # TODO Think whether threat subgraphs differently or not
        for subGraph in self.__subGraph:
            for tensor in subGraph:
                datum = {
                    'dtype': tensor['dtype'],
                    'scale': self.__save_np(tensor['scale']),
                    'zerop': self.__save_np(tensor['zerop']),
                    'quantized_dimension': tensor['quantized_dimension']
                }
                if type(tensor['value']) == numpy.ndarray:
                    datum['value'] = self.__save_np(tensor['value'])
                data[tensor['name']] = datum
        with open(self.__json_path, 'w') as json_file:
            json.dump(data, json_file)
