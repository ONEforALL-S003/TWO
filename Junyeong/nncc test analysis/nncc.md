# nncc test analysis

## 1. /one/nncc 파일 분석

```
#!/bin/bash

NNCC_SCRIPT_RPATH="scripts"
NNCC_COMMAND_RPATH="infra/nncc/command"

NNCC_PROJECT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNCC_SCRIPT_PATH="${NNCC_PROJECT_PATH}/${NNCC_SCRIPT_RPATH}"
```

변수 설정:

NNCC_SCRIPT_RPATH는 "scripts"로 설정됩니다.

NNCC_COMMAND_RPATH는 "infra/nncc/command"로 설정됩니다.

NNCC_PROJECT_PATH는 스크립트를 포함하는 디렉토리의 절대 경로로 설정됩니다.

NNCC_SCRIPT_PATH는 프로젝트 내의 "scripts" 디렉토리의 절대 경로로 설정됩니다.

```
function Usage()
{
  echo "Usage: $0 [COMMAND] ..."
  echo "Command:"
  for file in "$NNCC_COMMAND_RPATH"/*;
  do
    echo "  $(basename "$file")"
  done
}
```

Usage 함수:

Usage()는 스크립트를 사용하는 방법에 대한 정보를 제공하기 위해 정의된 함수입니다. "infra/nncc/command" 디렉토리의 파일에서 사용 가능한 명령을 나열합니다.

```

# Get command from command-line
COMMAND=$1; shift

if [[ -z "${COMMAND}" ]]; then
  Usage
  exit 255
fi
```

명령행에서 명령 가져오기:

스크립트는 사용자가 전달한 첫 번째 인수를 COMMAND 변수로 읽으려 시도합니다.
명령이 제공되지 않은 경우 (COMMAND가 비어 있음), Usage 함수를 호출하고 255의 종료 코드로 종료됩니다.

```

COMMAND_FILE="${NNCC_PROJECT_PATH}/${NNCC_COMMAND_RPATH}/${COMMAND}"

if [[ ! -f "${COMMAND_FILE}" ]]; then
  echo "ERROR: '${COMMAND}' is not supported"
  Usage
  exit 255
fi
```

명령 파일의 존재 여부 확인:

NNCC_PROJECT_PATH와 NNCC_COMMAND_RPATH를 사용하여 명령 파일의 경로를 생성합니다.
지정된 명령 파일이 존재하지 않는 경우, 오류 메시지를 표시하고 Usage 함수를 호출한 다음 255의 종료 코드로 종료합니다.

```

export NNCC_PROJECT_PATH
export NNCC_SCRIPT_PATH

```

변수 내보내기:

NNCC_PROJECT_PATH와 NNCC_SCRIPT_PATH를 환경 변수로 내보냅니다. 이러한 환경 변수는 이 스크립트에 의해 호출되는 다른 스크립트 또는 프로그램에서 액세스할 수 있습니다.

```

function import()
{
  source "${NNCC_PROJECT_PATH}/infra/nncc/config/$1"
}

```

import 함수:

import()는 "infra/nncc/config" 디렉토리에 있는 구성 파일을 소스(실행)하는 함수로 정의됩니다.

```

source "${COMMAND_FILE}" "$@"

```

명령 파일을 소스:

스크립트는 사용자의 입력에 지정된 명령 (COMMAND)과 스크립트에 제공된 추가 인수와 함께 해당 명령 파일을 소스(실행)합니다.
