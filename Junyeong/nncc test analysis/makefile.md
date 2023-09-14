# nncc test analysis

## 8. makefile

경로 : one/build/makefile

```makefile
# Target rules for targets named angkor_test

# Build rule for target.
angkor_test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 angkor_test
.PHONY : angkor_test

# fast build rule for target.
angkor_test/fast:
	$(MAKE) -f compiler/angkor/CMakeFiles/angkor_test.dir/build.make compiler/angkor/CMakeFiles/angkor_test.dir/build
.PHONY : angkor_test/fast
```

makefile에서 angkor_test 에 대한 파일을 생성할 때 빌드를 진행하는듯함.

하지만 이것은 빌드하는 부분이지 막상 명령어를 실행시켜 test 코드를 실행시키는 것이 아님. 다른 파일 찾을 필요 있음.
