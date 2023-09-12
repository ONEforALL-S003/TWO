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
