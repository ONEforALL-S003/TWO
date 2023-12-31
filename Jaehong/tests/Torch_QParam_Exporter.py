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

from Torch2CircleMapper import Torch2CircleMapper
from TorchExtractor import TorchExtractor


class TorchQParamExporter:
    def __init__(self, quantized_model :torch.nn.Module, json_path: str):
        if quantized_model is None or not isinstance(quantized_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model")
        if json_path is None:
            raise Exception("Please specify save path")
        
        idx = json_path.rfind(os.path.sep)
        if idx == -1:
            dir_path = ""
            json_file_name = json_path
        else:
            dir_path = json_path[:idx + 1]
            json_file_name = json_path[idx + 1:]

        self.__quantized_model = quantized_model

        self.__mapper = Torch2CircleMapper(dir_path)
        self.__extractor = TorchExtractor(dir_path, json_path, json_file_name)

        self.__mapping = None


    def set_mapping(self, original_model, sample_input):
        self.__mapping = self.__mapper.get_mapped_dict(original_model, sample_input)


    def save(self):
        tree = self.__extractor.extract_module(self.__quantized_model)
        mapping = self.__mapping
        self.__extractor.generate_files(tree, mapping)