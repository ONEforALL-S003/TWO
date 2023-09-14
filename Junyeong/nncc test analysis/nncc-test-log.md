# nncc test analysis

## 4. ./nncc test 를 실행시켰을 때 나오는 로그 분석

```
ssafy@ubuntu-20:/one$ ./nncc test
Test project /one/build
        Start   1: angkor_test
  1/117 Test   #1: angkor_test ................................   Passed    0.01 sec
        Start   2: arser_test
  2/117 Test   #2: arser_test .................................   Passed    0.01 sec
        Start   3: bino_test

        ...

115/117 Test #115: circle-interpreter-test ....................   Passed    0.22 sec
        Start 116: visq_unittest
116/117 Test #116: visq_unittest ..............................   Passed    0.30 sec
        Start 117: luci_pass_value_test
117/117 Test #117: luci_pass_value_test .......................   Passed   94.90 sec

100% tests passed, 0 tests failed out of 117

Total Test time (real) = 663.41 sec
```

단위테스트를 하는 것 같아 보입니다.
angkor_test 를 시작으로 luci_pass_value_test를 진행합니다.

총 1번부터 117번까지 있는 듯 합니다.

이 테스트 과정을 정의하는 파일을 찾아보면 될 것 같습니다.
