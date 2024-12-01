
import argparse
import numpy as np
import gym
import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseAgentConfig(ABC):
    """
    Base agent config
    """
    # from env
    device: str = "cpu" # 학습 장치 
    state_dim: int = 0 # 상태 차원
    action_dim: int = 0 # 액션 차원
    action_type: str = "" # 액션 타입

    @abstractmethod
    def environment(self, env: gym.Env | str):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.action_type == "discrete" else env.action_space.shape[0]
        self.action_type = "discrete" if isinstance(env.action_space, gym.spaces.Discrete) else "continuous"

    @abstractmethod
    def build(self):
        """
        build agent
        """
        pass
    
    @abstractmethod
    def training(self, actor_network: nn.Module, critic_network: nn.Module):
        """
        initialize agent from network
        """
        pass

    @abstractmethod
    def setup_networks(self, config: dict, args: argparse.Namespace = None):
        """
        setup networks
        """
        pass    
    
    @staticmethod
    def add_to_argparse(parser: argparse.ArgumentParser):
        pass


@dataclass
class BaseValueBasedAgent(ABC):
    """
    Base value-based agent
    """


@dataclass
class BaseActorCriticAgent(ABC):
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