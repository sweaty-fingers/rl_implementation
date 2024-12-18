# policy.py
## 기본 작동 원리
- policy network는 feature extractor를 통해 추출된 feature를 입력으로 받는다.
- feature extractor는 observation space의 특성을 고려하여 적절한 feature를 추출한다.
- feature extractor는 주로 CNN, MLP 등의 네트워크를 사용하여 이미지나 벡터 데이터를 처리한다.
- feature extractor는 추출된 feature를 적절한 형태로 변환하여 policy network에 전달한다.
- policy network는 이러한 feature를 입력으로 받아서 mlp layer를 통해 최종 출력을 계산한다.
- 이 출력은 각 행동에 대한 확률 또는 가치 추정치를 나타내며, 이를 통해 행동 선택이 이루어진다.

## 주요 메서드
- forward: 주어진 관측값에 대해 행동 추정치를 계산하는 메서드
- _predict: 주어진 관측값에 대해 최적의 행동을 추정하는 메서드
- _get_constructor_parameters: 모델을 재생성하는 데 필요한 매개변수를 반환하는 메서드
- device: 모델이 실행되는 장치를 반환하는 속성

## 주요 속성
- observation_space: 모델의 입력 공간을 나타내는 속성
- action_space: 모델의 출력 공간을 나타내는 속성
- features_extractor: 특징 추출기를 나타내는 속성
- features_dim: 특징 추출기의 출력 차원을 나타내는 속성
- net_arch: 모델의 네트워크 구조를 나타내는 속성
- activation_fn: 활성화 함수를 나타내는 속성
- normalize_images: 이미지 데이터를 정규화할지 여부를 나타내는 속성
