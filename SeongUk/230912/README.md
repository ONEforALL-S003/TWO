# quantized circle model (화)
Q-implant에 생성해주신 param을 넣어보려고 하였다.

### 발생한 에러들
1. 
```
terminate called after throwing an instance of 'std::runtime_error'
  what() : shape.size() == 1
```
npy를 저장하기전 data 검증 도입하여 해결

``` python
if data.shape == ():
    data = np.array([data])
```

2. 
```
line 132: zero point: ./out/Net_Conv2d/6.npy size: 1
values

<f4 |u1
{'descr': '<f4', 'fortran_order': False, 'shape': (4,), }                                                            

terminate called after throwing an instance of 'std::runtime_error'
  what():  formatting error: typestrings not matching
```

q-implant의 메서드 set_scale이나 set_zerop에서의 

```
npy::LoadArrayFromNumpy(scale_path, shape, fortran_order, scale);
```

을 실행하면 파일 /home/ssafy/ONE/ONE/externals/LIBNPY/include/npy.hpp 로 넘어가는데 
해당 파일의 580번 줄에서

```
  if (header.dtype.tie() != dtype.tie())
  {
    throw std::runtime_error("formatting error: typestrings not matching");
  }
```

data type이 맞지 않아서 생기는 에러 였다.

value에서의 타입이 안맞는 문제를 params의 bias의 value들을 전부 제거하여 해결하였다.
value이슈는 json생성시에 계산을 하지 않고, q-implant내에서 계산을 해주면 해결되지 않을까.

그 이후 weight.scale 의 dtype 형태가 안맞아서 float64 -> float 32로 변경 해주었다.