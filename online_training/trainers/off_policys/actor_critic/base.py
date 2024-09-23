import argparse
import torch
from colorama import Fore, Style, init 

from utils.managers import (
    ConfigManager, 
    log_decorator, 
    tensorboard_decorator, 
)

class BaseTrainer():
    """
    Base actor-critic schema
    """
    @tensorboard_decorator
    @log_decorator
    def __init__(self, args: argparse.Namespace = None, logger=None):
        config_manger = ConfigManager(args.config)
        self.config = config_manger.get_config()
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

    def add_log(self, msg, level="debug"):
        log_levels = ["debug", "info", "warning", "error", "critical"]
        if self.logger is None:
            print(msg)
        else:
            if level in log_levels:
                getattr(self.logger, level)(msg)
            else:
                print(f"Log level should be in {log_levels}")