from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import online_training.buffers.utils as util


class ReplayBuffer():
    """
    A simple FIFO experience replay buffer for SAC agents.
    from (https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py)
    """
    def __init__(self, obs_dim, act_dim, size):
        """
        obs, obs2: 현재 상태, 다음 스탭의 상태.
        ptr: 데이터를 삽입할 buffer 위치.
        """
        self.obs_buf = np.zeros(util.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(util.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(util.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        """
        Replay buffer에 데이터 저장.
        """
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size # max_size에 도달하면 첫번 째 요소로 돌아감.
        self.size = min(self.size+1, self.max_size) # max_size 이후 부터는 계속 max_size 유지

    def sample_batch(self, batch_size=32):
        """
        Buffer에 현재까지 채워져있는 샘플 중 batch_size 만큼의 데이터 추출
        """
        idxs = np.random.randint(0, self.size, size=batch_size) # [0, self.size) 사이의 batch_size 만큼의 indices 추출
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

