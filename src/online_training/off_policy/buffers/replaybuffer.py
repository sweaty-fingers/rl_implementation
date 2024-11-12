import argparse
import numpy as np
import torch
from online_training.off_policy.buffers.util import combined_shape

CONFIG_IGNORE = ["args"]
BUFFER_SIZE = 2000

class ReplayBuffer():
    _config = {
        "state_dim": None,
        "action_dim": None,
        "buffer_size": BUFFER_SIZE,
        "device": "cpu"
    }

    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, state_dim, action_dim, *args, buffer_size=BUFFER_SIZE, device="cpu", **kwargs):
        """
        env_config (dict): experience 들의 차원과 같은 정보들이 들어있는 dictionary
        """
        self._buffer_size = buffer_size
        self._device = device
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        # Set memory
        self._ptr, self._size = 0, 0
        self.states = np.zeros(combined_shape(self._buffer_size, self._state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(self._buffer_size, self._action_dim), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(self._buffer_size, self._state_dim), dtype=np.float32)
        self.rewards = np.zeros(self._buffer_size, dtype=np.float32)
        self.dones = np.zeros(self._buffer_size, dtype=np.float32)

    @property
    def config(self)->dict:
        """
        buffer config 반환
        """
        return self._config
    
    @classmethod
    def get_params(cls, config: dict, args: argparse.Namespace = None, ckpt=None):
        """
        buffer 인스턴스를 초기화하기 위한 parameters 반환
        """
        args = vars(args) if args is not None else {}
        if ckpt is None:
            cls._config = {
                "state_dim": config["env"]["state_dim"],
                "action_dim": config["env"]["action_dim"]
            }
        else:
            cls._config = config["buffer"]
        
        cls._config["device"] = args.get("device", "cpu"),
        cls._config["buffer_size"] = args.get("buffer_size", BUFFER_SIZE)

        return cls._config
    
    def __len__(self):
        """
        buffer 길이 반환 (실제 데이터가 들어간 양)
        """
        return self._size

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def add_experience(self, state, action, reward, next_state, done):
        """
        buffer에 experience(transition) 저장
        만약 experience들이 복수 (batch)로 들어올 경우 for문을 통해 하나씩 추가, 
        단일 경험이 경우 바로 추가.
        if isinstance(dones, list):
            assert isinstance(dones[0], list), "A done shouldn't be a list"
        """
        self.states[self._ptr] = state
        self.actions[self._ptr] = action
        self.next_states[self._ptr] = next_state
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = done
        self._ptr = (self._ptr+1) % self._buffer_size # buffer_size에 도달하면 첫번 째 요소로 돌아감.
        self._size = min(self._size+1, self._buffer_size) # buffer_size 이후 부터는 계속 max_size 유지

    def sample_batch(self, batch_size=None):
        """
        Buffer에 현재까지 채워져있는 샘플 중 batch_size 만큼의 데이터 추출
        """

        idxs = np.random.randint(0, self._size, size=batch_size) # [0, self.size) 사이의 batch_size 만큼의 indices 추출
        batch = {"states": self.states[idxs], 
                 "actions": self.actions[idxs], 
                 "next_states": self.next_states[idxs], 
                 "rewards": self.rewards[idxs], 
                 "dones": self.dones[idxs]
                 }
        
        return {k: self._to_tensor(v) for k, v in batch.items()}
    
    def __repr__(self) -> str:
        docs = ""
        for k, v in self.config.items():
            docs += f"{k}: {v}"
            docs += "\n"

        return docs
    
    @staticmethod
    def add_to_argparse(parser):
        """
        Replay buffer의 hyperparameters
        """
        parser.add_argument("--buffer_size", type=int, default=BUFFER_SIZE, help="Max size of replay buffer")
        return parser
    
if __name__ == "__main__":
    import gymnasium as gym
    from online_training.off_policy.buffers.util import _setup_parser
    
    parser = _setup_parser()
    config = {
        "env": {"state_dim": (40, 15), "action_dim": (1)}
    }

    parser = ReplayBuffer.add_to_argparse(parser)
    args = parser.parse_args()
    buffer = ReplayBuffer(**ReplayBuffer.get_params(config, args=args))

    print(buffer)

    
