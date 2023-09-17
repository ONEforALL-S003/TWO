# TIL

`./nncc test` 입력 시 스크립트 흐름

- `import` 함수는 `config` 경로에 있는 파일을 가져옴

- `infra/nncc/config/build.configuration` 안에는 build 후 결과물이 존재하는 경로가 `BUILD_WORKSPACE_RPATH` 변수로 선언되어 있다.

- `build` 경로가 없으면 먼저 빌드하라는 메세지 출력 후 255 종료.

- 그게 아니라면 `build` 경로로 이동한 후 `ctest` 실행.

- 실제 결과물을 확인해보면 `q-implant`도 cmake로 빌드 시 생성되는 `CTestTestfile.cmake`의 경로에 포함되어 있다.

- 문제는 간단해지는데, 어떻게 테스트가 추가되는지만 알면 된다.

[참고](http://freehuni.blogspot.com/2017/12/ctest.html)

- `luci-value-test`, `luci-pass-value-test`와 차이를 살펴보면서 테스트를 추가하려면 어떻게 하면 될지 살펴본다.

## add_test

자동으로 테스트를 실행하는 다른 경로의 `CmakeLists.txt`와 `q-implant`의 그것을 비교해보면 다른 곳에는 있고 q-implant에만 없는 것이 있는데, 바로 add_test 구문이다.

구조를 일반화하면 다음과 같다.

- `macro() ~ endmacro()`: 특정 문장을 sh 명령어로 치환하기 위한 매크로 선언부. include가 무조건 따라나온다.

- `include(test.lst)`: 테스트하려는 대상의 목록. 우리가 만들 q-implant에는 torch operator 목록이 들어갈 것이다.

    - `include(test.local.lst)`: 로컬 환경에서 테스트케이스를 커스텀으로 만들어 넣을 수 있다. 이걸 활용하면 이용자가 간단히 테스트케이스를 넣는 인터페이스를 구축할 수 있을 것 같다.

- `add_test(...)`: 실제 테스트가 추가되는 부분.

    - `NAME q_implant_test`: 테스트 이름을 지정

    - `COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/eval_impant.sh"`: 쉘 스크립트를 실행시킨다.

    - 위 규칙들을 통해 테스트를 추가할 수 있다. 일단은 `q_implant_export_test`부터 만들어야 한다.
    ```
    add_test(NAME q_implant_export-test # qparams를 생성해보는 테스트
        COMMAND "${CMAKE_CURRENT_SOURCE_DIR/eval_export.sh}"
                "${CMAKE_CURRENT_BINARY_DIR}"
                ...
    )
    ```

    - 세부적인 테스트 내용은 `eval_export.sh`에서 정의할 수 있다.

    - 쉘 스크립트 상에서 빌드 경로 상의 tflite2circle.exe에 접근할 수 있다면, exporter의 convert 부분을 아예 파이썬이 아닌 쉘에서 처리하게 한 후 파이썬 스크립트에는 양자화된 torch 경로, json 경로, 오리지널 torch 경로 및 이름을 매핑할 circle 경로 네 개를 넘기는 것도 생각해 볼 수 있다.
      