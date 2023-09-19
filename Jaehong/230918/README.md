# nncc test 추가하기

경로 상에 추가해야 할 것들

```
 q-implant
    |
    |
    \---- src
    |
    \---- tests
    |        |
    |        \---- *.py # 쉘 스크립트에서 실행할 파이썬 스크립트
    |        |
    |        \---- CMakeLists.txt # for ctest
    |        |
    |        \---- q_implant_export.sh  # 테스트 쉘 스크립트
    |
    \---- CMakeLists.txt # add_subdirectory
```

