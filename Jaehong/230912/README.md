# TIL

이슈 발생

- `Luci::CircleParam`의 내부 Tensor가 float형 vector를 받음

    -> float64 넣으면 터짐. 32짜리로 바꿔줘야 하는가?

- `q-implant` 112번째 줄 LoadArray에서 npy 파일이 32비트여야 함

```cpp
  // float64 가중치에 대해, 예외 발생
  npy::LoadArrayFromNumpy(scale_path, shape, fortran_order, scale);

  THROW_UNLESS(shape.size() == 1);
  THROW_UNLESS(fortran_order == false);

  // float32 필요
  node->quantparam()->scale = scale;
```