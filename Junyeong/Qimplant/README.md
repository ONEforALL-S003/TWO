# Qimplant

## 1. q-params 정의

- scale

- zerop

- value(weight)

-> 해당 목록을 .npy 로 빼내야함

## 2. 문제점

bias의 경우 bias에만 해당하는 quantization parameter 를 찾을 수 없음..

## 3. 샘플코드

우정님이 짜주신 q-params 추출 코드입니다.

- export_main.py

: 메인 기능을 하는 파일입니다. Torch_QParam_Exporter.py를 호출하여 q-params를 생성하고, Conv2d 모델을 tflite로 변환시킵니다.

- Torch_QParam_Exporter.py

: q-params 를 추출하는 코드입니다.

- Net_Conv2d.py, Net_Conv2d_3.py

: Conv2d 모델을 생성하는 파일입니다. Net_Conv2d.py의 경우 Net_Conv2d_3.py 의 모델을 import하여 사용합니다.

## 4. enhanced code

우정님 코드의 tflite 파일을 바로 circle로 변환할 수 있도록 subprocess 를 이용하여 tflite2circle을 실행할 수 있도록 하였습니다.

```python
import subprocess

# tflite2circle
tflite_filename = 'conv2d_original.tflite'
tflite_path = dir + tflite_filename
circle_filename = tflite_filename.replace('.tflite', '.circle')
output_path = dir + circle_filename

# execute tflite2circle file
try:
    subprocess.run(['./tflite2circle', tflite_path, output_path], check=True)
    print(f"{tflite_path} file is converted to {output_path}")
except subprocess.CalledProcessError:
    print("Error while converting tflite file")
```
