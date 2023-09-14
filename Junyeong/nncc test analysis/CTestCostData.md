# nncc test analysis

## 6. 이름을 저장하는 CTestCostData.txt

경로 : one/build/Testing/Temporary

```
angkor_test 5 0.00101689
arser_test 5 0.00157039
bino_test 5 0.000611346
cli_test 5 0.000646077
cwrap_test 5 0.000694968
fipe_test 5 0.000573798
hermes_test 5 0.0460668
kuma_test 5 0.000789771
mio_circle04_helper_test 5 0.000838787
mio_circle05_helper_test 5 0.000851199
mio_tf_test 5 0.00201137
mio_tflite260_helper_test 5 0.000663667
mio_tflite280_helper_test 5 0.000632923

...
```

실행시켜야 하는 테스트 파일의 이름을 담은 파일로 보입니다.
이후 두 숫자는 어떤걸 의미하는 것을 모르지만 argument로 보입니다.


--- 23.09.14 17:33 ---

참고 : https://ikcoo.tistory.com/218

GTEST의 경우 터미널에서 

[RUN___]
[____OK]

이런식으로 나오는데 해당 글에서 OK의 여부는 사전에 입력한 값과 Test함수에 넣고 나온 출력값이 같은지로 판단합니다.
따라서 위에 나온 test이름, 숫자는 특정 테스트의 argument로 작동하고 마지막에 나온 실수형의 수는 사전 입력값이라 생각합니다.

