# nncc test analysis

## 7. gtest_main.cc

5의 LastTest.log에서 로그를 분석하다가
Running main() from /one/externals/GTEST/googletest/src/gtest_main.cc

을 찾았습니다.

- gtest_main.cc

```cpp
#include <cstdio>
#include "gtest/gtest.h"

#if GTEST_OS_ESP8266 || GTEST_OS_ESP32
#if GTEST_OS_ESP8266
extern "C" {
#endif
void setup() {
  testing::InitGoogleTest();
}

void loop() { RUN_ALL_TESTS(); }

#if GTEST_OS_ESP8266
}
#endif

#else

GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

```

해당 코드는 플랫폼이 ESP8266, ESP32 인 경우 setup()과 loop() 함수를 쓰고, 아닐 경우 main() 함수를 쓴다는 내용입니다.

우리는 임베디드 보드 환경에서 돌리지 않기 때문에 main()으로 생각하면 되고 argc, argv 에 대해서 initGoogleTest를 진행합니다.

---

근데 이걸 불러서 테스트를 진행한다는건 알겠는데
어떤 코드로 gtest_main(argc, argv) 를 넣는지 찾아야합니다.
