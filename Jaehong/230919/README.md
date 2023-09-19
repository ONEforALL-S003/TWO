# 테스트 자동화에 대한 고찰

현재 python 모듈은 상당히 거대해져 있습니다. 파일 변환부터 qparams를 뽑아내는 것까지 다이렉트로 하는 건 좋은데, 각 단계가 일부 수정을 거칠 때마다 코드 전체를 바꿔야 하는 문제가 발생합니다.

테스트는 sh 스크립트 기반으로 돌아가고, 여기선 바깥 폴더에 있는 빌드 파일도 어렵지 않게 가져올 수 있습니다. 각 단계에 대해 스크립트 코드 한 줄만 추가, 삭제, 변경하는 것만으로도 쉽게 구조 수정이 가능합니다.

스크립트에 추가할 내용

- torch 모델과 각 포맷의 출력 경로를 받으면 양자화된 torch 모델, 원본 tflite 모델을 생성하는 py 모듈 실행

- tflite2circle 로 ftlite를 circle로 변환

- 양자화된 torch모델과 circle input, qparams.json 경로를 받으면 파일을 생성하는 py 모듈 실행

## q_extract_torch test

모듈을 분리하고, qparams를 생성하는 과정까지는 여기서 담당하게 합니다.

## q_implant test

여기선 좀 더 q-implant의 본질에 충실한 테스트만 돌릴 수 있도록 합니다.

- input.circle과 양식이 맞는 qparams를 받았을 때 성공적으로 output.circle을 만들어내는가?

- 양자화된 torch 또는 tensorflow 모델과 output.circle 간 추론이 비슷한가?

1번 테스트를 만들면서 어떤 언어로, 어떻게 돌릴지 정해야 합니다.

- python script로 돌릴 것인가? 아니면 gtest로 할 것인가.

- qparams를 별도로 생성할 것인가?

---

common-artifacts를 가져오면 같은  operator에 대해 circle을 가져올 수 있다?

- requirements.cmake에 common-artifacts를 추가한다.

---

성욱님 trial에서 res/.../PyTorchExamples를 가져와도 실제 python 스크립트에서 모듈을 제대로 가져오지 못하는 것을 봤습니다... 내일은 절대경로로 시도한다든가 해보면서 완성해봐야겠습니다. 어떻게 할지는 그려지는데, 자잘한 문법들이 괴롭히네요.