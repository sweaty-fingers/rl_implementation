import argparse
import numpy as np

from typing import Optional
from utilities.managers import (
    config_decorator,
    logger_decorator
)

OPTIMIZER_MODULE_NAME = "torch.optim"
LOSS_MODULE_NAME = "online_training.off_policys.actor_critic.losses"

N_STEP_LEARNING = 1
GAMMA = 0.99

class BaseAgent():
    """
    Base actor-critic schema
    """
    @config_decorator
    @logger_decorator
    def __init__(self, args: Optional[argparse.Namespace] = None, config=None, logger=None):
        self.logger = logger
        args = vars(args) if args is not None else {}
        
        # Envs config
        self.state_dim = config["env"]["state_dim"]
        self.action_type = config["env"]["action_type"]
        self.action_dim = config["env"]["action_dim"]

        self.checkpoint = config["checkpoint"]
        
        self.global_step_num = 0 # 전체 학습 step
        self.global_episode_num = 0 # 진행된 전체 episode number
        
        # Rewards
        self.n_step_learning = args.get("n_step_learning", N_STEP_LEARNING) # n_step td
        self.gamma = args.get("gamma", GAMMA)
        
    def add_log(self, msg, level="debug"):
        """
        log 메시지 추
        """
        log_levels = ["debug", "info", "warning", "error", "critical"]
        if self.logger is None:
            print(msg)
        else:
            if level in log_levels:
                getattr(self.logger, level)(msg)
            else:
                print(f"Log level should be in {log_levels}")
    
    @property
    def state_dict(self):
        pass

    def load_state_dict(self, ckpt):
        pass
                
    @staticmethod
    def add_to_argparse(parser):
        """
        Parser에 classa argument 추가
        """

        parser.add_argument(
            "--gpus",
            default=None,
            type=int,
            help="id(s) for GPU_VISIBLE_DEVICES(MPS or CUDA)",
        )
        parser.add_argument(
            "--optimizer_module",
            type=str,
            default=OPTIMIZER_MODULE_NAME,
            help="optimizer module to get optimizer class"
        )
        parser.add_argument(
            "--optimizer",
            type=str,
            default=None,
            help="optimizer class from torch.optim",
        )

        parser.add_argument("--lr_policy", type=float, default=None)
        parser.add_argument("--lr_critic", type=float, default=None)
        parser.add_argument("--gamma", type=float, default=GAMMA, help="Discount factor gamma of reward")
        parser.add_argument(
            '--n_step_learning',
            type=int,
            default=N_STEP_LEARNING,
            help="N_steps to get td error"
        )
        return parser 
