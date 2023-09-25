# MobileNet

- Menual 한 방식으로 qp를 만들어서 q-implant를 적용해보자

## Resource

ONE/runtime/contrib/TFLiteSharp/TFLiteTestApp/res/mobilenet_v1_1.0_224.tflite

- 기존에 사용하고 있는 mobilenet v1 을 기반으로 테스트 진행.

## tflite2circle

- 코드로 사용하면 subprocess를 쓰겠지만, 우선 구조를 비교하기 위해 circle로 변경

- tflite2circle 경로

  - ONE/build/compiler/tflite2circle/tflite2circle

- 임시 저장 경로

  - ONE/compiler/q-implant/tests/test_output/MN_input.circle

- command

```
/one/runtime/contrib/TFLiteSharp/TFLiteTestApp/res/mobilenet_v1_1.0_224.tflite /one/compiler/q-implant/tests/test_output/MN_input.circle
```

-> DepthwiseConv2D와 Conv2D로 대부분 레이어가 구성되어있음

Conv2D filter bias Relu6 형식

circle도 모양과 형식은 같음.

![image](https://github.com/ONEforALL-S003/TWO/assets/79979086/45a29f4d-e334-4510-97e5-e726cf2c336f)

![image](https://github.com/ONEforALL-S003/TWO/assets/79979086/b47a262d-b424-4949-a37c-1b843eb8b797)


netron 상에서 Conv2D는 filter, bias 차원 일치.

but DepthwiseConv2D의 OP name은 같은데,

output -> 0

input -> 1

filter -> 2

bias -> 3

으로 변환되어서 나옴.

형식 보기위해 circledump 진행

- command

```
./one/build/compier/circledump/circledump /one/compiler/q-implant/tests/test_output/MN_input.circle
```

결과물은 [mobilenet_circle_log.txt](https://github.com/ONEforALL-S003/TWO/blob/main/Junyeong/MobileNet/mobilenet_circle_log.txt)https://github.com/ONEforALL-S003/TWO/blob/main/Junyeong/MobileNet/mobilenet_circle_log.txt 에 저장.
