# 자주 발생하는 에러

- ### json에서 circle에 없는 name이 존재할경우 발생하는 에러
```
terminate called after throwing an instance of 'std::out_of_range'
  what():  _Map_base::at
```

- ### numpy.array 의 형태가 스칼라일경우 발생하는 에러
```
terminate called after throwing an instance of 'std::runtime_error'
  what() : shape.size() == 1
```

- ### numpy.array 에서의 dtype과 circle tensor의 dtype이 맞지 않을때 발생하는 에러
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  formatting error: typestrings not matching
```

- ### circle에 존재하는 노드이나 qparam.json에 name이 없을경우 발생하는 에러
```
terminate called after throwing an instance of 'std::runtime_error'
  what():  from_node->quantparam()
```