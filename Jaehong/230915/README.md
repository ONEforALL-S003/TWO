# 23.09.15 작업 내용

- Exporter 클래스가 너무 비대해짐

    -> Extractor와 Mapper로 분리할 필요가 생김

    - Mapper: Torch 모델을 Circle 모델과 매핑하여 다시 이름을 붙인 Dict 반환
    - Extractor: Circle로 매핑된 Dict에서 실제로 qparams를 생성