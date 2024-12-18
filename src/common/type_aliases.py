"""Common aliases for type hints"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Optional, Protocol, SupportsFloat, Union

import gymnasium as gym
import numpy as np
import torch as th

# Avoid circular imports, we use type hint as string to avoid it too
if TYPE_CHECKING: # 타입 체크 툴 (mypy or Pyright)를 사용할 때만 TYPE_CHECKING이 True가 되어 import
    from src.common.callbacks import BaseCallback
    from src.common.vec_env import VecEnv

GymEnv = Union[gym.Env, "VecEnv"]
GymObs = Union[tuple, dict[str, Any], np.ndarray, int]
GymResetReturn = tuple[GymObs, dict]
AtariResetReturn = tuple[np.ndarray, dict[str, Any]]
GymStepReturn = tuple[GymObs, float, bool, bool, dict]
AtariStepReturn = tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]
TensorDict = dict[str, th.Tensor]
OptimizerStateDict = dict[str, Any]
MaybeCallback = Union[None, Callable, list["BaseCallback"], "BaseCallback"]
PyTorchObs = Union[th.Tensor, TensorDict]

# A schedule takes the remaining progress as input
# and outputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class DictRolloutBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: th.Tensor
    next_observations: TensorDict
    dones: th.Tensor
    rewards: th.Tensor


class RolloutReturn(NamedTuple):
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"


class PolicyPredictor(Protocol): # 프로토콜은 타입 힌트를 제공하는 데 사용되며, 클래스나 함수가 특정 인터페이스를 구현하도록 강제하는 데 사용된다.
# 즉 Protocol 타입힌트를 만족하려면 클래스에 predict 메서드를 구현해야 한다. 
    def predict(
        self,
        observation: Union[np.ndarray, dict[str, np.ndarray]],
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, Optional[tuple[np.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """