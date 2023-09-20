# extract test 모듈 작성

일단 별도의 브랜치를 파고, extract가 잘 되는지 단위 테스트를 추가해보기로 한다.

extract의 실행 결과가 커버하는 op의 수가 많지 않아서 테스트를 계속 만들지 고민해봐야 할듯

---

python 절대경로를 설정해서 파일을 가져올 수 있는가?

- `"$(dirname "${BASH_SOURCE[0]}")"`: `VERIFY_SOURCE_PATH`에 할당되는 현재 위치의 절대경로.

```sh
cat > "${TEST_RESULT_FILE}.log" <(
    exec 2>&1
    set -ex
    source "${VIRTUALENV}/bin/activate"
    "${VIRTUALENV}/bin/python" "--dir ${VERIFY_SOURCE_PATH}" "${VERIFY_SCRIPT_PATH}"
    if [[ $? -eq 0 ]]; then
      touch "${PASSED_TAG}"
    fi
  )
```

```py
# argparser 활용, 현재 절대경로를 인자로 받기
```