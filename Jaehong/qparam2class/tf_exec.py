import os

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib
from tf_param_fb_exporter import TfFbParamExporter


tflite_models_dir = pathlib.Path("resources/")
tflite_models_dir.mkdir(exist_ok=True, parents=True)
tflite_model_file = tflite_models_dir / "Add_000.tflite"
model = tflite_model_file.read_bytes()


exporter = TfFbParamExporter(model_content=model, json_path="out/Add_000/qparam.json")

exporter.save()