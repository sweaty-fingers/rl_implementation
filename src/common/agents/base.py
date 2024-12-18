
import argparse
import numpy as np
import gym
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseAgent(ABC):
    """
    Base agent
    """ 


@dataclass
class BaseValueBasedAgent(BaseAgent):
    """
    Base value-based agent
    """

    @abstractmethod
    def loss(self):
        """
        loss 계산
        """
        pass

    @abstractmethod
    def update(self, batch):
        """
        loss를 계산하고 네트워크를 한번 업데이트
        """
        pass

    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action


@dataclass
class BaseActorCriticAgent(BaseAgent):
    """
    Base actor-critic schema
    """
    # config
    config_path: str = "" # path of config file

    # optimizer
    optimizer_module: str = 'torch.optim' # optimizer module to get optimizer class
    optimizer: str = 'Adam' # optimizer class from torch.optim
    lr_actor: float = 1e-4 # learning rate of actor
    lr_critic: float = 1e-4 # learning rate of critic
    
    # Rewards
    n_step_learning: int = 1 # n_step td 아직 구현 x
    gamma: float = 0.99 # 할인 계수
    
    # training
    global_step_num: int = 0 # 전체 학습 step
    global_episode_num: int = 0 # 진행된 전체 episode number

    def __post_init__(self):
        self.from_config()
    
    @abstractmethod
    def from_args(self, args: argparse.Namespace):
        """
        initialize agent from args
        from config 뒤에 호출되어 config로 초기화된 인자들을 덮어씌움
        """
        for attr in ['optimizer', 'optimizer_module', 'lr_actor', 'lr_critic']:
            if (value := getattr(args, attr)) is not None:
                setattr(self, attr, value)
    
    @abstractmethod
    def from_config(self):
        """
        initialize agent from config file
        """
        if self.config_path:
            config = load_config(self.config_path)
            for attr in ['optimizer', 'optimizer_module', 'lr_actor', 'lr_critic']:
                setattr(self, attr, config.get(attr, getattr(self, attr)))
    
    @abstractmethod
    def setup(self, args: argparse.Namespace = None):
        """
        setup agent
        """
        if args.config_path:
            self.config_path = args.config_path

        self.from_config()
        self.from_args(args)
        pass

    @abstractmethod
    def state_dict(self):
        """
        return agent state
        """
        pass

    @abstractmethod
    def load_state_dict(self, ckpt):
        """
        load agent state
        """
        pass

    @abstractmethod
    def update(self, batch):
        """
        update agent
        """
        pass
    
    @abstractmethod
    def critic_loss(self):
        """
        return critic loss
        """
        pass
    
    @abstractmethod
    def policy_loss(self):
        """
        return policy loss
        """
        pass
    
    @abstractmethod
    def get_action_and_action_info(self, state: torch.Tensor):
        """
        return action and action info
        """
        pass
    
    @abstractmethod
    def sample_action(self, state, is_eval=False):
        """
        return sampled action
        """
        pass
    
    @staticmethod
    def add_to_argparse(parser):
        """
        Parser에 classa argument 추가
        """
        parser.add_argument(
            "--optimizer",
            help="optimizer class from torch.optim",
        )
        parser.add_argument("--lr_actor", type=float, help="learning rate of actor")
        parser.add_argument("--lr_critic", type=float, help="learning rate of critic")
        parser.add_argument("--gamma", type=float, help="Discount factor gamma of reward")
        parser.add_argument(
            '--n_step_learning',
            type=int,
            help="N_steps to get td error"
        )
        parser.add_argument("--config_path", type=str, help="path of config file")
        return parser