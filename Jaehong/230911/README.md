# TIL

- torch 모델 양자화 시 QuantStub, DequantStub을 앞뒤로 감싸줘야 함.

- QuantStub은 입력을 양자화해줄 때의 고유한 값을 가져야 하므로 특정 상태를 갖는다.

- DequantStub은 특별한 상태를 갖지 않음. 고유할 필요가 없으므로 어떤 값을 갖지도 않는다.

[참고](https://discuss.pytorch.org/t/what-do-de-quantstub-actually-do/82402/2)

