# luci-value-test-분석

## luci-value-test

우선 readme.md 내용입니다.

```
luci-value-test validates luci IR graph model file (.circle)

The test proceeds as follows

Step 1: Generate tflite files and circle files from TFLite recipes (listsed in test.lst).

"TFLite recipe" -> tflchef -> "tflite file" -> tflite2circle -> "circle file"
Step 2: Run TFLite interpreter and luci-interpreter for the generated tflite and circle, respectively. (with the same input tensors filled with random values)

circle file -> luci-interpreter -------> Execution result 1
tflite file -> TFLite interpreter -----> Execution result 2
Step 3: Compare the execution result 1 and 2. The result must be the same.
```

요약하자면

1. tflite recipe를 이용해 tflite 파일을 만들고 tflite2circle을 이용해 circle 파일도 만듭니다.

2. 랜덤 input tensor를 만들어 TFLite 인터프리터와 luci 인터프리터를 각각 돌립니다.

3. result 값을 비교합니다. 같아야합니다.

즉 이 테스트는 추론의 결과값을 비교하는 과정으로 tflite -> circle 변화가 잘 이루어졌는지 확인하는 테스트입니다.

- CMakeLists.txt

```CMake
if(NOT ENABLE_TEST)
  return()
endif(NOT ENABLE_TEST)

unset(LUCI_VALUE_TESTS)
unset(LUCI_VALUE_TESTS_TOL)

macro(addeval NAME)
  list(APPEND LUCI_VALUE_TESTS ${NAME})
endmacro(addeval)

macro(addevaltol NAME RTOL ATOL)
  list(APPEND LUCI_VALUE_TESTS_TOL ${NAME} ${RTOL} ${ATOL})
endmacro(addevaltol)

# Read "test.lst"
include("test.lst")
# Read "test.local.lst" if exists
include("test.local.lst" OPTIONAL)
```

test.lst에는 tflite recipes 가 들어있습니다.
이것을 불러옵니다.

```CMake
# Generate dependencies
add_custom_target(luci_eval_testfiles ALL DEPENDS ${TESTFILES})

if(NOT CMAKE_CROSSCOMPILING)

  get_target_property(ARTIFACTS_BIN_PATH testDataGenerator BINARY_DIR)

  add_test(NAME luci_value_test
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify.sh"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${ARTIFACTS_BIN_PATH}"
            "${NNCC_OVERLAY_DIR}/venv_2_12_1"
            "$<TARGET_FILE:luci_eval_driver>"
            ${LUCI_VALUE_TESTS}
  )
```

add_test를 통해 luci_value_test 라는 이름의 테스트를 추가하고

eververify.sh 를 실행시키는데 여러가지 argument를 넣습니다.

```CMAKE

if(DEFINED LUCI_VALUE_TESTS_TOL)
  add_test(NAME luci_value_tol_test
    COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverifytol.sh"
            "${CMAKE_CURRENT_BINARY_DIR}"
            "${ARTIFACTS_BIN_PATH}"
            "${NNCC_OVERLAY_DIR}/venv_2_12_1"
            "$<TARGET_FILE:luci_eval_driver>"
            ${LUCI_VALUE_TESTS_TOL}
  )
endif()

else(NOT CMAKE_CROSSCOMPILING)
# NOTE target test is carried out using reference input/output data from host
#      test results. this is because it would be difficult to prepare
#      TensorFlow lite for target device.
#      thus, one must run the host test and then run the test in target device
#      with the test result files from the host test.

if(NOT DEFINED ENV{BUILD_HOST_EXEC})
  message(STATUS "BUILD_HOST_EXEC not set: Skip luci-value-test")
  return()
endif(NOT DEFINED ENV{BUILD_HOST_EXEC})

set(ARTIFACTS_BIN_PATH $ENV{BUILD_HOST_EXEC}/compiler/common-artifacts)

add_test(NAME luci_value_cross_test
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverify_ref.sh"
          "${CMAKE_CURRENT_BINARY_DIR}"
          "${ARTIFACTS_BIN_PATH}"
          "$<TARGET_FILE:luci_eval_driver>"
          ${LUCI_VALUE_TESTS}
)

if(DEFINED LUCI_VALUE_TESTS_TOL)
  add_test(NAME luci_value_cross_tol_test
           COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/evalverifytol_ref.sh"
                   "${CMAKE_CURRENT_BINARY_DIR}"
                   "${ARTIFACTS_BIN_PATH}"
                   "$<TARGET_FILE:luci_eval_driver>"
                   ${LUCI_VALUE_TESTS_TOL}
  )
endif()

endif(NOT CMAKE_CROSSCOMPILING)

```

나머지는 마찬가지의 과정으로, 테스트 명을 정의하고 스크립트 파일을 실행하여 테스트를 진행합니다.

그럼 evalverify.sh, evalverifytol.sh, evalverify_ref.sh, evalverifytol_ref.sh 에 대해 알아보겠습니다.

해당 파일은 모두

`"${CMAKE_CURRENT_BINARY_DIR}"`

에 있는 파일 입니다.

이후 계속..
