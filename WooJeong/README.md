[check.py](./check.py)  
one/res/PyTorchExamples/examples 폴더를 check.py와 같은 경로로 넣고 실행하면 torch -> onnx -> tf -> tflite 변환 과정에서 실패하는 Operator들을 확인 가능  

Tensorflow는 기본적으로 NHWC(배치 사이즈, 높이, 너비, 채널)을 지원하나, Torch/Onnx 는 NCHW(배치 사이즈, 채널, 높이, 너비)를 지원해서  
Convolution과 같은 연산이 torch/onnx에서 tflite로 변환 되면 tflite 상에 traspose operation이 들어가게 됨  
![onnx](https://github.com/ONEforALL-S003/TWO/assets/136890801/59d0e539-3f04-4fb2-9bb3-f65126a94f2c)  
![tflite](https://github.com/ONEforALL-S003/TWO/assets/136890801/ca127206-91bc-4878-843b-7d7b5f27841b)  
![q-implant](https://github.com/Samsung/ONE/issues/11254) Issue 상으로는 OHWI 이라 되어 있는 것 같은데 compatible with NHWC format 이라 되어 있음으로  
qparam을 뽑을 때 tensor(numpy)를 NHWC format으로 변환해서 저장할 필요성이 존재
