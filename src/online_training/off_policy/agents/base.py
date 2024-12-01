from abc import ABC, abstractmethod
from dataclasses import dataclass
import argparse
import gym

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