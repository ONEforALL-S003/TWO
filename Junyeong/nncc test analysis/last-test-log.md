# nncc test analysis

## 5. 테스트 로그 발견

파일 위치 : one/build/Testing/Temporary/LastTest.log

테스트 파일을 실행시킬 때 중간에 나오는 출력을 저장한 결과물로 보입니다.

```
Start testing: Sep 14 13:19 KST
----------------------------------------------------------
1/117 Testing: angkor_test
1/117 Test: angkor_test
Command: "/one/build/compiler/angkor/angkor_test"
Directory: /one/build/compiler/angkor
"angkor_test" start time: Sep 14 13:19 KST
Output:
----------------------------------------------------------
Running main() from /one/externals/GTEST/googletest/src/gtest_main.cc
[==========] Running 88 tests from 22 test suites.
[----------] Global test environment set-up.
[----------] 2 tests from ADT_FEATURE_BUFFER
[ RUN      ] ADT_FEATURE_BUFFER.ctor
[       OK ] ADT_FEATURE_BUFFER.ctor (0 ms)
[ RUN      ] ADT_FEATURE_BUFFER.access
[       OK ] ADT_FEATURE_BUFFER.access (0 ms)
[----------] 2 tests from ADT_FEATURE_BUFFER (0 ms total)

[----------] 3 tests from ADT_FEATURE_CHW_LAYOUT
[ RUN      ] ADT_FEATURE_CHW_LAYOUT.col_increase
[       OK ] ADT_FEATURE_CHW_LAYOUT.col_increase (0 ms)
[ RUN      ] ADT_FEATURE_CHW_LAYOUT.row_increase
[       OK ] ADT_FEATURE_CHW_LAYOUT.row_increase (0 ms)
[ RUN      ] ADT_FEATURE_CHW_LAYOUT.ch_increase
[       OK ] ADT_FEATURE_CHW_LAYOUT.ch_increase (0 ms)
[----------] 3 tests from ADT_FEATURE_CHW_LAYOUT (0 ms total)

[----------] 3 tests from ADT_FEATURE_HWC_LAYOUT
[ RUN      ] ADT_FEATURE_HWC_LAYOUT.C_increase
[       OK ] ADT_FEATURE_HWC_LAYOUT.C_increase (0 ms)
[ RUN      ] ADT_FEATURE_HWC_LAYOUT.W_increase
[       OK ] ADT_FEATURE_HWC_LAYOUT.W_increase (0 ms)
[ RUN      ] ADT_FEATURE_HWC_LAYOUT.H_increase
[       OK ] ADT_FEATURE_HWC_LAYOUT.H_increase (0 ms)
[----------] 3 tests from ADT_FEATURE_HWC_LAYOUT (0 ms total)

...

[----------] 8 tests from TensorShapeTest
[ RUN      ] TensorShapeTest.ctor
[       OK ] TensorShapeTest.ctor (0 ms)
[ RUN      ] TensorShapeTest.ctor_initializer_list
[       OK ] TensorShapeTest.ctor_initializer_list (0 ms)
[ RUN      ] TensorShapeTest.resize
[       OK ] TensorShapeTest.resize (0 ms)
[ RUN      ] TensorShapeTest.dim
[       OK ] TensorShapeTest.dim (0 ms)
[ RUN      ] TensorShapeTest.copy
[       OK ] TensorShapeTest.copy (0 ms)
[ RUN      ] TensorShapeTest.eq_negative_on_unmatched_rank
[       OK ] TensorShapeTest.eq_negative_on_unmatched_rank (0 ms)
[ RUN      ] TensorShapeTest.eq_negative_on_unmatched_dim
[       OK ] TensorShapeTest.eq_negative_on_unmatched_dim (0 ms)
[ RUN      ] TensorShapeTest.eq_positive
[       OK ] TensorShapeTest.eq_positive (0 ms)
[----------] 8 tests from TensorShapeTest (0 ms total)

[----------] Global test environment tear-down
[==========] 88 tests from 22 test suites ran. (0 ms total)
[  PASSED  ] 88 tests.
<end of output>
Test time =   0.01 sec
----------------------------------------------------------
Test Passed.
"angkor_test" end time: Sep 14 13:19 KST
"angkor_test" time elapsed: 00:00:00
----------------------------------------------------------

2/117 Testing: arser_test
2/117 Test: arser_test
Command: "/one/build/compiler/arser/arser_test"
Directory: /one/build/compiler/arser
"arser_test" start time: Sep 14 13:19 KST
Output:

...

```

이런식으로 테스트 과정에서 발생한 로그를 저장하는 파일로 보입니다.

Command를 보면 어떤 테스트 파일을 실횅시키는지 나와있습니다.

해당 디렉토리로 가서 실행파일을 실행시키면 (ex : ./angkor_test) log에 저장되어 있는 내역과 동일한 내용이 터미널에 출력됩니다.

따라서 테스트 하는 모듈의 이름을 담는 파일을 import 해서 반복문을 돌며 유닛 테스트를 진행 -> "> LastTest.log" 이런식으로 로그를 저장하는듯 합니다.
