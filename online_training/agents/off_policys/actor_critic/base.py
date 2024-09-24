import argparse
import numpy as np
import torch

from colorama import Fore, Style, init
from typing import Optional
from online_training.trainers.metrics import MetricsMeter

from utils.managers import (
    config_decorator,
    logger_decorator
)

POSSIBLE_CRITERION = ["min", "max"]

OPTIMIZER_MODULE_NAME = "torch.optim"
LOSS_MODULE_NAME = "online_training.off_policys.actor_critic.losses"

STEPS_PER_EPOCH = 2000
STEPS_PER_EPOCH_VALID = 100
EPOCHS_PER_SAVE = 1000

class BaseAgent():
    """
    Base actor-critic schema
    """
    @config_decorator
    @logger_decorator
    def __init__(self, args: Optional[argparse.Namespace] = None, config=None, logger=None):
        self.config = config
        self.logger = logger
        self.args = vars(args) if args is not None else {}
        # Set Device
        self.device = "cpu"
        self.gpus = self.args.get("gpus", None)
        if self.gpus is not None:
            if torch.cuda.is_available():
                self.device = f"cuda:{self.gpus}"
            elif torch.backends.mps.is_available():
                self.device = f"mps:{self.gpus}"

        self.add_log(f"{Fore.GREEN}{self.device}{Style.RESET_ALL}", level="info")
        
        # Envs
        self.action_type = self.config["action_type"]
        self.action_dim = self.config["action_dim"]
        
        # Metric
        self.metrics = MetricsMeter()
        self.test_metric = self.args.get("test_metric", None)
        self.criterion = self.args.get("test_criterion", None)
        if self.criterion not in POSSIBLE_CRITERION:
            raise ValueError(f"criterion should be in {POSSIBLE_CRITERION}\n Now get {self.criterion}")

        if self.criterion == "min":
            self.best_metric = np.inf
        elif self.criterion == "max":
            self.best_metric = -np.inf
       
    def add_log(self, msg, level="debug"):
        log_levels = ["debug", "info", "warning", "error", "critical"]
        if self.logger is None:
            print(msg)
        else:
            if level in log_levels:
                getattr(self.logger, level)(msg)
            else:
                print(f"Log level should be in {log_levels}")
                
    @staticmethod
    def add_to_argparse(parser):

        parser.add_argument(
            "--gpus",
            default=None,
            type=int,
            help="id(s) for GPU_VISIBLE_DEVICES(MPS or CUDA)",
        )
        parser.add_argument(
            "--epochs_per_save", type=int, default=EPOCHS_PER_SAVE, help="epochs per saving model state"
        )
        parser.add_argument(
            '--steps_per_epoch',
            type=int,
            default=STEPS_PER_EPOCH,
            help="Max step nubmer for one epoch"
        )
        parser.add_argument(
            '--steps_per_epoch_valid',
            type=int,
            default=STEPS_PER_EPOCH,
            help="Max step nubmer for one epoch"
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
        parser.add_argument(
            "--test_metric",
            type=str,
            default=None,
            help="Test metric to choose the best model"
        )
        parser.add_argument(
            "--criterion",
            type=str,
            default=None,
            help="Min or Max for comparing test metric"

        )
        return parser 
