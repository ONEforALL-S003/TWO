import os

import torch
import torch.nn
import torch.quantization

from Torch2CircleMapper import Torch2CircleMapper
from TorchExtractor import TorchExtractor


# Helper class of PyTorch Quantization Parameter Export
class TorchQParamExporter:
    def __init__(self, original_model: torch.nn.Module, quantized_model: torch.nn.Module,
                 sample_input: torch.tensor, json_path: str, tflite2circle_path='./tflite2circle'):
        if quantized_model is None or not isinstance(original_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model")
        if quantized_model is None or not isinstance(quantized_model, torch.nn.Module):
            raise Exception("There is no Pytorch Model")
        if json_path is None:
            raise Exception("Please specify save path")
        if sample_input is None or not isinstance(sample_input, torch.Tensor):
            raise Exception("Please give sample input of network")
        if not os.path.exists(tflite2circle_path):
            raise Exception('tflite2circle not exists')

        idx = json_path.rfind(os.path.sep)
        if idx == -1:
            dir_path = ""
        else:
            dir_path = json_path[:idx + 1]

        mapper = Torch2CircleMapper(original_model=original_model, sample_input=sample_input,
                                    dir_path=dir_path, tflite2circle_path=tflite2circle_path)
        self.__mapping, partial_graph_data = mapper.get_mapped_dict()
        self.__extractor = TorchExtractor(quantized_model=quantized_model, json_path=json_path,
                                          partial_graph_data=partial_graph_data)

    def save(self):
        self.__extractor.generate_files(self.__mapping)