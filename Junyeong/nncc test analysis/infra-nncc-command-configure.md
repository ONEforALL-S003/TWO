# nncc test analysis

## 3. /one/infra/nncc/command/configure 파일 분석

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

mkdir -p "${BUILD_WORKSPACE_PATH}"

```

디렉토리 생성:

mkdir -p "${BUILD_WORKSPACE_PATH}" 명령을 사용하여 BUILD_WORKSPACE_PATH 디렉토리를 생성합니다. -p 옵션은 이미 디렉토리가 존재하는 경우에도 오류 없이 생성하도록 합니다.

```

cd "${BUILD_WORKSPACE_PATH}"
cmake "${NNCC_PROJECT_PATH}/infra/nncc" "$@"

```

작업 디렉토리로 이동 및 CMake 실행:

cd "${BUILD_WORKSPACE_PATH}" 명령을 사용하여 작업 디렉토리로 이동합니다.
그런 다음 cmake 명령을 사용하여 CMake 빌드 시스템을 설정합니다. 이 명령은 주어진 CMake 프로젝트 디렉토리와 추가 인수 ($@)를 사용하여 빌드 환경을 설정합니다.

---

여기서도 마찬가지로 BUILD_WORKSPACE_PATH 를 지정할 때 NNCC_PROJECT_PATH 와 BUILD_WORKSPACE_RPATH 를 사용합니다.

하지만 BUILD_WORKSPACE_RPATH 가 안나왔습니다..

아무래도 이 부분은 build.configuration 에서 import 하여 사용한다고 추측할 수 밖에 없을 것 같습니다. 그런데 해당 경로의 파일을 찾을 수 없어서.. 일단은 넘어가겠습니다.
