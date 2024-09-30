import argparse
import itertools
import numpy as np
import torch

from typing import Optional
from utilities.buffer_util import combined_shape
from utilities.util import make_config

CONFIG_IGNORE = ["args"]
BUFFER_SIZE = 2000

class ReplayBuffer():
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, config: dict, args: argparse.Namespace = None):
        """
        env_config (dict): experience 들의 차원과 같은 정보들이 들어있는 dictionary
        """
        self.args = vars(args) if args is not None else {}
        self.buffer_size = args.get("buffer_size", BUFFER_SIZE)
        self.batch_size = config["trainer"]["batch_size"]
        self.device = config["trainer"]["device"]
        self.state_dim = config["env"]["state_dim"]
        self.action_dim = config["env"]["action_dim"]

        self._config = make_config(self, CONFIG_IGNORE)
        
        # Set memory
        self.ptr, self.size = 0, 0
        self.states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros(combined_shape(self.buffer_size, self.action_dim), dtype=np.float32)
        self.next_states = np.zeros(combined_shape(self.buffer_size, self.state_dim), dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)

    @property
    def config(self):
        return self._config
    
    def __len__(self):
        return self.size

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self.device)

    def add_experience(self, state, action, reward, next_state, done):
        """
        buffer에 experience(transition) 저장
        만약 experience들이 복수 (batch)로 들어올 경우 for문을 통해 하나씩 추가, 
        단일 경험이 경우 바로 추가.
        if isinstance(dones, list):
            assert isinstance(dones[0], list), "A done shouldn't be a list"
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr+1) % self.buffer_size # buffer_size에 도달하면 첫번 째 요소로 돌아감.
        self.size = min(self.size+1, self.buffer_size) # buffer_size 이후 부터는 계속 max_size 유지

    def sample_batch(self, force_batch_size=None):
        """
        Buffer에 현재까지 채워져있는 샘플 중 batch_size 만큼의 데이터 추출
        """
        if force_batch_size is not None:
            batch_size = force_batch_size
        else:
            batch_size = self.batch_size

        idxs = np.random.randint(0, self.size, size=batch_size) # [0, self.size) 사이의 batch_size 만큼의 indices 추출
        batch = {"state": self.states[idxs], 
                 "action": self.actions[idxs], 
                 "next_state": self.next_states[idxs], 
                 "rewards": self.rewards[idxs], 
                 " dones": self.dones[idxs]
                 }
        
        return {k: self._to_tensor(v) for k, v in batch.items()}
    
    @staticmethod
    def add_to_argparse(parser):
        """
        Replay buffer의 hyperparameters
        """
        parser.add_argument("--buffer_size", type=int, default=BUFFER_SIZE, help="Max size of replay buffer")
        return parser