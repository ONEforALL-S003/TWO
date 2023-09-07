import os
import torch
import onnx
import onnx_tf
import tensorflow as tf
import importlib

from pathlib import Path

examples = os.listdir('examples')
output_folder = "./output/"
Path(output_folder).mkdir(parents=True, exist_ok=True)
success = []
fail = []
fail1 = []
fail2 = []
fail3 = []
len = len(examples)
cnt = 0
etype = 0
for example in examples:
    try:
        print(example)
        etype = 0
        module = importlib.import_module("examples." + example)
        # save .pth
        torch.save(module._model_, output_folder + example + ".pth")

        etype = 1
        opset_version = 9
        if hasattr(module._model_, 'onnx_opset_version'):
            opset_version = module._model_.onnx_opset_version()

        onnx_model_path = output_folder + example + ".onnx"

        torch.onnx.export(
            module._model_, module._dummy_, onnx_model_path, opset_version=opset_version)

        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
        onnx.checker.check_model(inferred_model)
        onnx.save(inferred_model, onnx_model_path)

        etype = 2
        tf_prep = onnx_tf.backend.prepare(inferred_model)
        tf_prep.export_graph(path=output_folder + example + ".TF")

        etype = 3
        # for testing...
        converter = tf.lite.TFLiteConverter.from_saved_model(output_folder + example + ".TF")
        converter.allow_custom_ops = True
        converter.experimental_new_converter = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

        tflite_model = converter.convert()
        success.append(example)
    except:
        if etype == 0:
            fail.append(example)
        elif etype == 1:
            fail1.append(example)
        elif etype == 2:
            fail2.append(example)
        elif etype == 3:
            fail3.append(example)
    finally:
        cnt = cnt + 1
        print("{0} / {1}".format(cnt, len))

print("success : {0}".format(success))
print("unexpected : {0}".format(fail))
print("onnx fail : {0}".format(fail1))
print("tf fail : {0}".format(fail2))
print("tflite fail : {0}".format(fail3))
