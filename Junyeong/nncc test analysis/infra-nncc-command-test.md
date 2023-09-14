# nncc test analysis

## 2. /one/infra/nncc/command/test 파일 분석

```
#!/bin/bash

import "build.configuration"

```

import 구문:

import "build.configuration"을 사용하여 "build.configuration" 파일을 소스(실행)합니다. 이 파일은 프로젝트 빌드와 관련된 구성 정보를 포함하고 있을 것으로 추정됩니다.

```

BUILD_WORKSPACE_PATH="${NNCC_PROJECT_PATH}/${BUILD_WORKSPACE_RPATH}"

```

BUILD_WORKSPACE_PATH 설정:

BUILD_WORKSPACE_PATH 변수는 NNCC_PROJECT_PATH와 BUILD_WORKSPACE_RPATH를 조합하여 프로젝트의 작업 디렉토리 경로를 설정합니다.

```

if [[ ! -d "${BUILD_WORKSPACE_PATH}" ]]; then
  echo "'${BUILD_WORKSPACE_RPATH}' does not exist. Please run 'configure' first"
  exit 255
fi

```

작업 디렉토리 확인:

BUILD_WORKSPACE_PATH가 디렉토리로 존재하지 않는 경우, "'BUILD_WORKSPACE_RPATH'가 존재하지 않습니다. 먼저 'configure'를 실행하십시오"라는 오류 메시지를 출력하고 종료 코드 255로 스크립트를 종료합니다. 이것은 작업 디렉토리가 빌드하기 전에 설정되어야 함을 나타냅니다.

```

export CTEST_OUTPUT_ON_FAILURE=1

```

환경 변수 설정:

CTEST_OUTPUT_ON_FAILURE 환경 변수를 1로 설정합니다. 이렇게 하면 빌드 실패 시 테스트 출력이 표시됩니다.

```

cd "${BUILD_WORKSPACE_PATH}" && ctest "$@"

```

작업 디렉토리로 이동 및 ctest 실행:

BUILD_WORKSPACE_PATH로 디렉토리를 변경한 다음 ctest 명령을 실행합니다. $@를 사용하여 스크립트에 전달된 모든 인수를 ctest 명령으로 전달합니다. 이는 프로젝트 빌드와 관련된 테스트를 실행하는 부분으로 보입니다

---

## 분석 결과

BUILD_WORKSPACE_PATH 는 NNCC_PROJECT_PATH 와 BUILD_WORKSPACE_RPATH 의 조합으로 경로가 이루어지는데 NNCC_PROJECT_PATH 는 nncc 파일을 실행할 때 나오지만 BUILD_WORKSPACE_RPATH 는 나온게 없어서 export 하는 부분을 찾아야함. configure 파일이나 build 파일에 있을 것으로 추정
