# DQN 구현
## Initialization q_net and q_net_target
dqn에서는 target network가 behavior network과 동일한 모델을 사용하므로 동일한 가중치를 사용.

```python
self.q_net = self.make_q_net()
self.q_net_target = self.make_q_net()
self.q_net_target.load_state_dict(self.q_net.state_dict()) 
```

## Attributes
### class attributes
- exploration_schedule: random action을 취할 확률을 정하는 함수.
- q_net: behavior network. 
- q_net_target: target network. 
- policy: q_net과 q_net_target을 사용하여 액션을 선택하는 정책. 

### instance attributes
- exploration_initial_eps [0, 1], (float) : 학습 초기에 새로운 state 로의 탐색을 위해 random 액션을 취할 확률.

- exploration_final_eps [0, 1], (float) : 학습이 진행됨에 따라 random 액션을 취할 확률을 점차 낮추게 됨. 이때 마지막 탐색 확률을 설정하는 값.(threshold)

- exploration_fraction[0, 1], (float) : 학습이 진행됨에 따라 random action을 취할 확률을 점차 낮추는 비율을 설정하는 값.

- target_update_interval, (int) : target update 주기를 설정하는 값. 환경 스텝 단위로 설정. 여러 환경을 동시에 학습할 경우 환경 단위로 설정할 수 있음. 각 환경 전체의 step을 계산.

- _n_calls, (int) : _on_step() 메서드가 호출된 횟수.

- max_grad_norm, (float) : gradient clipping의 maximum 값.
- exploration_rate, (float) : 현재 random action을 취할 확률.
- _init_setup_model, (bool) : instance 생성 시 모델을 초기화할지 여부를 설정하는 값.



