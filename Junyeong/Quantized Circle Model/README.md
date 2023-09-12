# Quantized Circle Model

## Qimplant를 통해 Quantized Circle Model을 생성

이전 과정을 통해 qparam.json 과 x.npy, input.circle이 갖춰졌습니다.

이제 멘토님이 제공해주신 skeleton code 를 바탕으로 생성한 q-implant 를 적용해 보겠습니다.

1. one/build/compiler/q-implant 의 q-implant 실행파일을 가져옵니다.

2. export_main.py 의 마지막에 해당 내용을 추가합니다.

```python
# execute q-implant

# q-implant [input.circle] [qparam.json] [output.circle file name]

# In the current state, I am hardcoding the paths,
# but I plan to adjust them according to the actual paths
# in the future.

output_circle = "output.circle"
try:
    result = subprocess.run(['./q-implant', output_path, "./export/Net_Conv2d/qparam.json", output_circle], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        print(f"q-implant has successfully executed, and {output_circle} has been generated.")
    else:
        print(f"q-implant failed with the following error:\n{result.stderr}")
except subprocess.CalledProcessError as e:
    print(f"Error while executing q-implant. Error message:\n{e.stderr}")
```

3. 결과를 확인합니다.

```
Error while executing q-implant. Error message:
terminate called after throwing an instance of 'std::runtime_error'
  what():  tensor.isMember("quantized_dimension")
```

=> 멘토님이 짜준 스켈레톤 코드에는 quantized_dimension 이 key로 있었는데 여기선 안 넣어줘서 생긴 문제인 것 같습니다.

4. Torch_QParam_Exporter.py 수정

```python
# data[name] 의 마지막에 추가
'quantized_dimension': 0
```

5. 결과 확인

```
Error while executing q-implant. Error message:
terminate called after throwing an instance of 'std::out_of_range'
  what():  _Map_base::at
```

=> 해당 에러는 C++에서 map을 이용하여 name을 매칭하다 발생한 에러라고 합니다.

=> input.circle 과 qparam.json의 name, key가 일치하지 않아서 발생한 문제로 보입니다....

### conv2d_original.circle

![image](https://github.com/ONEforALL-S003/TWO/assets/79979086/5cabf36d-c78e-4cf9-a6ca-dd895500a05e)


### qparam.json

```json
{
  "quant": {
    "scale": "0.npy",
    "zerop": "1.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "conv1.weight": {
    "scale": "2.npy",
    "zerop": "3.npy",
    "value": "4.npy",
    "dtype": "int8"
  },
  "conv1": {
    "scale": "5.npy",
    "zerop": "6.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "conv1.bias": {
    "scale": "5.npy",
    "zerop": "6.npy",
    "dtype": "uint8",
    "quantized_dimension": 0,
    "value": "7.npy"
  },
  "conv2.weight": {
    "scale": "8.npy",
    "zerop": "9.npy",
    "value": "10.npy",
    "dtype": "int8"
  },
  "conv2": {
    "scale": "11.npy",
    "zerop": "12.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "conv2.bias": {
    "scale": "11.npy",
    "zerop": "12.npy",
    "dtype": "uint8",
    "quantized_dimension": 0,
    "value": "13.npy"
  },
  "conv3.weight": {
    "scale": "14.npy",
    "zerop": "15.npy",
    "value": "16.npy",
    "dtype": "int8"
  },
  "conv3": {
    "scale": "17.npy",
    "zerop": "18.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "conv3.bias": {
    "scale": "17.npy",
    "zerop": "18.npy",
    "dtype": "uint8",
    "quantized_dimension": 0,
    "value": "19.npy"
  },
  "subnet.conv1.weight": {
    "scale": "20.npy",
    "zerop": "21.npy",
    "value": "22.npy",
    "dtype": "int8"
  },
  "subnet.conv1": {
    "scale": "23.npy",
    "zerop": "24.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "subnet.conv1.bias": {
    "scale": "23.npy",
    "zerop": "24.npy",
    "dtype": "uint8",
    "quantized_dimension": 0,
    "value": "25.npy"
  },
  "subnet.conv2.weight": {
    "scale": "26.npy",
    "zerop": "27.npy",
    "value": "28.npy",
    "dtype": "int8"
  },
  "subnet.conv2": {
    "scale": "29.npy",
    "zerop": "30.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "subnet.conv2.bias": {
    "scale": "29.npy",
    "zerop": "30.npy",
    "dtype": "uint8",
    "quantized_dimension": 0,
    "value": "31.npy"
  },
  "subnet.conv3.weight": {
    "scale": "32.npy",
    "zerop": "33.npy",
    "value": "34.npy",
    "dtype": "int8"
  },
  "subnet.conv3": {
    "scale": "35.npy",
    "zerop": "36.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "subnet.conv3.bias": {
    "scale": "35.npy",
    "zerop": "36.npy",
    "dtype": "uint8",
    "quantized_dimension": 0,
    "value": "37.npy"
  },
  "subnet.conv4.weight": {
    "scale": "38.npy",
    "zerop": "39.npy",
    "value": "40.npy",
    "dtype": "int8"
  },
  "subnet.conv4": {
    "scale": "41.npy",
    "zerop": "42.npy",
    "dtype": "uint8",
    "quantized_dimension": 0
  },
  "subnet.conv4.bias": {
    "scale": "41.npy",
    "zerop": "42.npy",
    "dtype": "uint8",
    "quantized_dimension": 0,
    "value": "43.npy"
  }
}
```

