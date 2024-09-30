import argparse
import glob
import numpy as np
import torch


from utilities.buffer_util import combined_shape
from utilities.util import make_config, find_files

from typing import Optional, Dict, List
TensorBatch = List[torch.Tensor]

CONFIG_IGNORE = ["args"]
BUFFER_SIZE = 2000


class ReplayBuffer:
    """Replay buffer for offline reinforcement learning"""
    def __init__(
        self, state_dim, action_dim, batch_size, buffer_size: Optional[int] = None, device: str = "cpu", args: Optional[argparse.Namespace] = None, **kwargs):
        args = vars(args) if args is not None else {}
        self.buffer_size = args.get("buffer_size", BUFFER_SIZE) if buffer_size is None else buffer_size
        self._batch_size = batch_size
        self._state_dim = state_dim
        self._action_dim = action_dim
        
        self._config = make_config(self, CONFIG_IGNORE)

        self._ptr, self._size = 0, 0
        self._states = torch.zeros(
            combined_shape(buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._actions = torch.zeros(
            combined_shape(buffer_size, action_dim), dtype=torch.float32, device=device
        )
        self._rewards = torch.zeros(combined_shape(buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros(
            combined_shape(buffer_size, state_dim), dtype=torch.float32, device=device
        )
        self._dones = torch.zeros(combined_shape(buffer_size, 1), dtype=torch.float32, device=device)
        self._device = device

    def get_init_param_from_config(self, config: dict):
        kwargs = {}
        kwargs["_batch_size"] = config.get("trainer").get("batch_size")
        kwargs["_state_dim"] = config.get("env").get("state_dim")
        kwargs["_action_dim"] = config.get("env").get("action_dim")

        return kwargs

    @property
    def config(self):
        return self._config

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def reset_buffer(self):
        """
        Buffer 초기화
        """
        self._ptr, self._size = 0, 0
        self._states = torch.zeros(
            combined_shape(self._buffer_size, self._state_dim), dtype=torch.float32, device=self._device
        )
        self._actions = torch.zeros(
            combined_shape(self._buffer_size, self._action_dim), dtype=torch.float32, device=self._device
        )
        self._rewards = torch.zeros(combined_shape(self._buffer_size, 1), dtype=torch.float32, device=self._device)
        self._next_states = torch.zeros(
            combined_shape(self._buffer_size, self._state_dim), dtype=torch.float32, device=self._device
        )
        self._dones = torch.zeros(combined_shape(self._buffer_size, 1), dtype=torch.float32, device=self._device)
    
    def find_dirs(base_dir, pattern, extension):
        """
        base_dir 아래에 있는 디렉터리들 중 pattern을 만족하는 디렉터리들의 경로 반환
        """
        import os
        
        dir_list = []
        for root, dirs, files in os.walk(base_dir):
            if pattern in os.path.basename(root):
                for file in files:
                    if file.endswith(extension):                
                        dir_list.append(root)
                        break
        return dir_list        
                        
        
    def load_custom_dataset(self, filepath):
        """
        filepath 하위에 있는 특정 확장자를 만족하는 데이터 파일에서 일정 수 만큼의 데이터를 추출하여 buffer 구성.
        """
        self.reset_buffer()
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        
        __file__
        
        
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
              
        
        
        return 

    # Loads data in d4rl format, i.e. from Dict[str, np.array].
    def load_d4rl_dataset(self, data: Dict[str, np.ndarray]):
        if self._size != 0:
            raise ValueError("Trying to load data into non-empty replay buffer")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError(
                "Replay buffer is smaller than the dataset you are trying to load!"
            )
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._size += n_transitions
        self._ptr = min(self._size, n_transitions)

        print(f"Dataset size: {n_transitions}")

    def sample(self, batch_size: int) -> TensorBatch:
        indices = np.random.randint(0, min(self._size, self._ptr), size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_experience(self):
        # Use this method to add new data into the replay buffer during fine-tuning.
        # I left it unimplemented since now we do not do fine-tuning.
        raise NotImplementedError