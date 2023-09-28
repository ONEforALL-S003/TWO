# Op Level Test

`pad` 연산은 별도의 가중치를 갖지 않는다.

    단, 입력된 자료의 크기를 맞춰주는 연산의 특성 상 크기를 갖고 있어야 하므로 정수형의 텐서를 갖고 있다. 이건 양자화의 대상이 아니므로 q-implant에서는 건너뛰는 것이 맞다.

# QImplant.write 메서드 개선

- 메서드는 분리해서 따로 쓰는 게 좋겠다.

- traversal은 이미 만들어진 걸 쓰는 게 좋다.

    - back propagation이 안 된다?

- circle tensor dump를 참조하여 value test를 실행해본다?

- h5로 저장되는 파일에서 값을 읽어들일 수 있다

- compare_tensors.py 참조하여 값을 비교해볼 것

