import collections
import copy
import os
import shutil
import sys

import numpy as np
import onnx
import onnx_tf
import tensorflow as tf
import torch
import torch.nn
import torch.quantization

sys.path.append('./include')
import subprocess

#  generated by pics.
#  we need to set dependency on cmakelist
from include.circle.Model import Model
from include.circle.SubGraph import SubGraph


class Torch2CircleMapper:
    @staticmethod
    def permute(tensor: torch.Tensor) -> torch.Tensor:
        dim = len(tensor.shape)
        if dim == 4:  # NCHW to NHWC
            tensor = tensor.permute(0, 2, 3, 1)
        return tensor

    def __init__(self, original_model: torch.nn.Module, sample_input: torch.Tensor, dir_path: str,
                 tflite2circle_path='./tflite2circle'):
        self.__dir_path = dir_path

        self.__mapping = None
        self.__reverse_mapping = None
        self.__network_input = None
        self.__network_output = None

        if not os.path.exists(tflite2circle_path):
            raise Exception('tflite2circle not exists')
        self.__tflite2circle_path = tflite2circle_path
        self.__original_model = original_model
        self.__sample_input = sample_input
        self.__partial_graph_data = collections.OrderedDict()

    def get_mapped_dict(self):
        if self.__mapping is not None:
            return self.__mapping, self.__partial_graph_data
        original_model = self.__original_model
        sample_input = self.__sample_input
        tmp_dir = os.path.join(self.__dir_path, "tmp")
        copied = False
        tries = 0

        # When there are same tensor(same shape and shape value), collision occur and mapping fails
        # try to map with different tensor values (uniformly rand value)
        while True:
            try:
                circle = self.__torch2circle(original_model, sample_input, tmp_dir)
                self.__generate_mapped_dict(circle)
                break
            except Exception:
                tries += 1
                if tries >= 3:
                    raise Exception('Failed to mapping')
                if not copied:
                    copied = True
                    original_model = copy.deepcopy(original_model)
                    # TODO: set tensor data uniformly random data to re-try mapping

        return self.__mapping, self.__partial_graph_data

    def __torch2circle(self, original_model: torch.nn.Module, sample_input: torch.Tensor, dir_path: str) -> Model:
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
        circle_path = os.path.join(self.__dir_path, 'input.circle')
        try:
            #  TODO: Need to set relative path of build of tflite2circle or get path by program argument
            subprocess.run([self.__tflite2circle_path, tflite_path, circle_path], check=True)
        except Exception:
            print('Fail to convert to circle')
        buf = bytearray(open(circle_path, 'rb').read())
        return Model.GetRootAsModel(buf)

    def __generate_mapped_dict(self, circle):
        # mapping torch name to circle name (key: torch name, value : circle name)
        # eg) torch: conv1.weight ->  circle: convolution;PartitionedCall/convolution
        self.__mapping = {}
        # mapping for circle tensor hash value to torch name (key: hashed circle tensor value,  value: torch name)
        # It uses Tensor value(numpy binary data including shape and value).
        # So when the tensors are unique, the key will be unique
        self.__reverse_mapping = reverse_mapping = {}
        original_model = self.__original_model
        sample_input = self.__sample_input

        if original_model is None or not isinstance(original_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model for mapping")
        if sample_input is None or not isinstance(sample_input, torch.Tensor):
            raise Exception("Please give sample input to convert model")

        params = original_model.named_parameters()

        # generate mapping data of original model's parameter
        for name, param in params:
            tensor = param.data
            tensor = self.permute(tensor)  # permute tensor if needed(To make equivalent of circle's)
            key = hash(tensor.numpy().tobytes())  # calculate hash value of binary numpy data
            if key in reverse_mapping:
                raise Exception('Duplicate Tensors exist')
            reverse_mapping[key] = name  # tensor hash value -> torch name

        self.__network_input = []
        for idx in range(circle.SubgraphsLength()):
            self.__circle_subgraph_mapping_traverse(circle, circle.Subgraphs(idx))
        shutil.rmtree(os.path.join(self.__dir_path, "tmp"))

        input_list = []
        output_list = []
        prev_module_name = None
        for name, mod in original_model.named_modules():
            if name == '':  # it's just model itself
                continue
            class_name = str(type(mod))
            if isinstance(mod, torch.quantization.QuantStub):
                input_list.append(name)
            elif isinstance(mod, torch.quantization.DeQuantStub):
                output_list.append(name)
            # TODO: find better way to check class in torch.nn.modules.activation package
            elif class_name.find('activation') != -1:
                # activation such as RELU, don't have tensor. So it can't be mapped
                # use previous operator data to map it
                if name not in self.__partial_graph_data:
                    self.__partial_graph_data[name] = {}
                self.__partial_graph_data[name]['prev_op'] = prev_module_name
            prev_module_name = name

        if len(input_list) == 1 and len(self.__network_input) == 1:
            self.__mapping[input_list[0]] = self.__network_input[0].Name().decode('utf-8')
        else:
            print("There are more than one input of Network. Please map it manually")

    def __circle_subgraph_mapping_traverse(self, circle: Model, graph: SubGraph):
        mapping, reverse_mapping = self.__mapping, self.__reverse_mapping
        # For operators those not have value
        op_mapping = {}

        # get input tensor of graph
        for idx in range(graph.InputsLength()):
            input_tensor = graph.Tensors(graph.Inputs(idx))
            self.__network_input.append(input_tensor)

        # get all of tensors from graph
        for idx in range(graph.TensorsLength()):
            tensor = graph.Tensors(idx)
            name = tensor.Name().decode('utf-8')
            shape = tensor.ShapeAsNumpy()
            # When the tensor don't have shape, We can't map it due to lack of tensor value
            if shape.size == 0:
                continue
            buffer = circle.Buffers(tensor.Buffer()).DataAsNumpy()
            # When fetched buffer is not type of numpy or size is 0 => The tensor actually have no value
            if type(buffer) is not np.ndarray or buffer.size == 0:
                continue
            key = hash(buffer.tobytes())

            # If equivalent torch tensor of current circle tensor, we can map it
            if key in reverse_mapping:
                origin_name = reverse_mapping[key]  # torch's name
                mapping[origin_name] = name  # mapping torch name to circle tensor name
                op_name = origin_name[:origin_name.rfind(".")]

                # To map tensor's those whom don't have tensor value, memorize tensor data(buffer index)
                if op_name not in op_mapping:
                    op_mapping[op_name] = set()
                op_mapping[op_name].add(idx)

        # approximately it takes O(N^2)
        # we need to think to it better way or not
        for i in range(graph.OperatorsLength()):
            operator = graph.Operators(i)
            input_set = set(operator.InputsAsNumpy().tolist())  # get operator's input tensor's indexes

            for op_name, op_input in op_mapping.items():
                # When there is subset of already mapped tensor's indexes
                # That mapped subset operator information is same with current operation
                # Then we can map torch operator name to circle's operator name
                if input_set.issuperset(op_input):
                    input_set = input_set - op_input
                    for tensor_idx in input_set:
                        tensor = graph.Tensors(tensor_idx)
                        tensor_name = tensor.Name().decode('utf-8')
                        mapping[op_name] = tensor_name  # torch operator name -> circle operator name

                    # can mapping output because it has only one!
                    if operator.OutputsLength() == 1:
                        output_tensor = graph.Tensors(operator.Outputs(0))
                        output_tensor_name = output_tensor.Name().decode('utf-8')
                        mapping[op_name + '.out'] = output_tensor_name
                    break
