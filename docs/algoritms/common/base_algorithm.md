# BaseAlgorithm


## attributes
policy_kwargs: policy_class에 전달되는 인자.
num_timesteps: 현재까지 학습한 스텝 수.
_total_timesteps: 총 학습 스텝 수.
_num_timesteps_at_start: 학습 시작 시점의 스텝 수.
start_time: 학습 시작 시점의 시간.
seed: random seed.
tensorboard_log: tensorboard 로그 경로. _logger를 초기화 할 때 인자로 들어감.
_last_obs: 마지막 관측 값
_last_original_obs: 전처리(normalize) 되지 않은 마지막 관측 값
_last_episode_starts: 어디에 사용되는지 확인하기

ep_info_buffer: episode 정보를 저장하는 버퍼?
ep_success_buffer: 성공했는지 여부를 저장하는 버퍼?

_vec_normalize_env: VecNormalize 객체

_supported_action_spaces: 

## methods
_setup_learn(): learn 매서드 초반에 호출되는 함수.
학습에 필요한 변수들을 초기화하고, 학습 준비를 마무리. total_timesteps, callback을 반환.

_get_policy_from_name(): class attribute으로 policy_aliases를 알고리즘마다 정해두고,
만약 policy 인자가 문자열이면 policy_aliases에서 해당 문자열을 찾아서 해당 클래스를 반환.






learning rate에 어떤 값이 들어가는지 확인하기
