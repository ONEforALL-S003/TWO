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
