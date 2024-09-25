"""Basic LightningModules on which other modules can be built."""
import argparse
import os
import shutil
import torch
import numpy as np

from tqdm import tqdm
from typing import Union
from colorama import Fore, Style, init

from utils.util import remove_old_checkpoints, get_attr_from_module
from utils.managers import (
    ConfigManager, 
    logger_decorator, 
    tensorboard_decorator, 
    update_tensorboard
)
from online_training.trainers.decorators import (
    add_to_csv, 
    update_metrics, 
    log_outputs, 
    save_checkpoint,
    update_scheduler
)

from online_training.trainers.metrics import MetricsMeter

init(autoreset=True)

OPTIMIZER_MODULE_NAME = "torch.optim"
OPTIMIZER = "Adam"
LR = 2e-5

NO_PROGRESS = False
CONFIG_IGNORE = ['training_device', 'data_config', 'lr', 'optimizer_class', 'loss_fn', 'scheduler']
FUNC_KEYS = ["loss_fn", "optimizer_class"]
TEST_METRIC = "loss"
CRITERION = "min" 
POSSIBLE_CRITERION = ["min", "max"]
STEPS_PER_EPOCH = 2000
STEPS_PER_EPOCH_VALID = 100
EPOCHS_PER_SAVE = 1000


class BaseTrainer():
    """
    Trainer schema
    """
    @tensorboard_decorator
    @logger_decorator
    def __init__(self, args: argparse.Namespace = None, logger=None, checkpoint=None, tb_writer=None):
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
        
        # DataModule & Model
        self.data_config = self.model.data_config
        self.input_dim = self.data_config["input_dim"]
        self.output_dim = self.data_config["output_dim"]
        
        # optimizer & scheduler
        self.optimizer = None
        optimizer_module = self.args.get("optimizer_module", OPTIMIZER_MODULE_NAME)
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = get_attr_from_module(optimizer_module, optimizer)
        self.lr = self.args.get("lr", LR)
        self.scheduler = self.args.get("scheduler_mode", None)

        loss_module = self.args.get("loss_module", LOSS_MODULE_NAME)
        loss = self.args.get("loss", LOSS)
        self.loss_fn = get_attr_from_module(loss_module, loss)
        self.add_log(f"{Fore.GREEN}Loss fuction: {self.loss_fn.__name__}{Style.RESET_ALL}", level="info")

        # Add model graphs in tensorboard
        if tb_writer is not None:
            self.add_log(f"{Fore.GREEN}Add model graph in tensorboard{Style.RESET_ALL}", level="info")
            dummy = torch.randn(self.data_config["input_dim"])
            tb_writer.add_graph(self.model, dummy)

        # Checkpoints
        self.save_dirs = self.config["CHECKPOINTS"]
        self.checkpoint = checkpoint # ckpt file (.pth)
        self.best_state = None
        
        # Metric
        self.metrics = MetricsMeter()
        self.test_metric = self.args.get("test_metric", TEST_METRIC)
        self.criterion = self.args.get("test_criterion", CRITERION)
        if self.criterion not in POSSIBLE_CRITERION:
            raise ValueError(f"criterion should be in {POSSIBLE_CRITERION}\n Now get {self.criterion}")

        if self.criterion == "min":
            self.best_metric = np.inf
        elif self.criterion == "max":
            self.best_metric = -np.inf
        
        # Training setting
        self.stage = "train"
        self.start_epoch = 0
        self.current_epoch = self.start_epoch
        self.step = 0
        self.steps_per_epoch = self.args.get("steps_per_epoch", STEPS_PER_EPOCH)
        self.steps_per_epoch_valid = self.args.get("steps_per_epoch_valid", STEPS_PER_EPOCH_VALID)
        self.epochs_per_save = self.args.get("epochs_per_save", EPOCHS_PER_SAVE)
        self.best_metric: Union[int, float]
        self.test_metric: str
        
        # reward
        self.success_reward: float
        self.fail_reward: float
        self.step_reward: float
        self.timeout_reward: float
        
        # Etc
        self.no_progress = self.args.get("no_progress", NO_PROGRESS)
    
    def configure_optimizers(self):
        """
        actor와 critic의 optimizer 정의
        """
        pass
    def forward(self, x):
        self.model(x)
    
    def predict(self, x):
        pass
    
    def fit(self, datamodule = None, max_epoch: int = 1) -> dict:
        """전체 학습 루프 작성"""
        self.models_to_device()
        self.configure_optimizers()
        
        if self.checkpoint:
            self.load_optimizer() 
            
        self.add_log("Training started....", "debug")
        for epoch in tqdm(range(self.start_epoch, max_epoch), initial=self.start_epoch, total=max_epoch, leave=False, position=0, desc=f"{Fore.BLUE}Start: fit, device: {self.device}{Style.RESET_ALL}"):
            self.current_epoch = epoch
            self.add_log(f"Epoch: {epoch}", "debug")

            train_outputs = self.training_epoch(
                dataloader=datamodule.train_dataloader(),
            )

            valid_outputs = self.validation_epoch(
                dataloader=datamodule.valid_dataloader()
            )

        outputs = {
            "train": train_outputs,
            "valid": valid_outputs,
            "epoch": self.current_epoch,
        }  #
        self.add_log("Training ended....", "debug")

        return outputs
    
    @add_to_csv("train_metric")
    @update_tensorboard
    @log_outputs("debug")
    def training_epoch(self, dataloader: torch.utils.data.dataloader = None):
        "training epoch 작업"
        self.stage = "train"
        self.metrics.reset_all()

        progress_bar = tqdm(total=self.steps_per_epoch, position=1, desc=f"{Fore.BLUE}Start: {self.stage}, device: {self.device}{Style.RESET_ALL}", leave=False)
        data_iter, batch_idx = iter(dataloader), 0
        while batch_idx < self.steps_per_epoch:
            try:
                batch = next(data_iter)
                _ = self.training_step(batch=batch, batch_idx=batch_idx)
                
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                _ = self.training_step(batch=batch, batch_idx=batch_idx)
            
            batch_idx += 1
            progress_bar.update(1)
            
        self.add_log(f"step: {batch_idx} -> epoch{self.current_epoch}", "debug") 
        outputs = {}
        for key, value in self.metrics.avg_items():
            outputs[key] = value
        
        outputs["lr"] = self.optimizer.param_groups[0]["lr"]
        return outputs
    
    @update_metrics
    def training_step(self, batch, batch_idx) -> dict:
        self.step += 1
        self.optimizer.zero_grad()
        outputs = self._run_on_batch(batch=self.data_to_device(batch))
        outputs["loss"].backward()
        self.optimizer.step()
        
        return outputs

    @update_scheduler("loss")
    @log_outputs("debug")
    @add_to_csv("valid_metric")
    @update_tensorboard
    @save_checkpoint
    def validation_epoch(self, dataloader: torch.utils.data.dataloader = None) -> dict:
        "validation epoch 작업"
        self.stage = "valid"
        self.metrics.reset_all()
        self.models_to_eval()
        
        progress_bar = tqdm(total=self.steps_per_epoch_valid, position=1, desc=f"{Fore.BLUE}Start: {self.stage}, device: {self.device}{Style.RESET_ALL}", leave=False)
        
        data_iter, batch_idx = iter(dataloader), 0
        while batch_idx < self.steps_per_epoch_valid:
            try:
                batch = next(data_iter)
                with torch.no_grad():
                    outputs = self.validation_step(batch=batch, batch_idx=batch_idx)

            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
                with torch.no_grad():
                    outputs = self.validation_step(batch=batch, batch_idx=batch_idx)

            progress_bar.update(1)
            batch_idx += 1
            
        outputs = {}
        for key, value in self.metrics.avg_items():
            outputs[key] = value
        
        return outputs
    
    @update_metrics
    def validation_step(self, batch, batch_idx):
        "validation step 작업"
        outputs = self._run_on_batch(batch=self.data_to_device(batch))
        return outputs
    
    def test(self) -> dict:
        """테스트 루프 작성"""
       
    def test_epoch(self, dataloader: torch.utils.data.dataloader = None, epoch_idx=1) -> dict:
        """test epoch 작업"""
        pass
    
    def test_step(self, batch, batch_idx):
        """est step 작업"""
        pass
    
    def _run_on_batch(self, batch, with_preds=False):
        """batch 데이터를 받아서 모델과 수행할 작업"""
        pass
    
    def data_to_device(self, batch):
        data_in_device = []
        for b in batch:
            data_in_device.append(b.to(self.device))
            
        return data_in_device
    
    def models_to_device(self):
        """target, pollicy model등 알고리즘에 맞게 추가하여 오버라이드"""
        self.model.eval()

    def models_to_eval(self):
        """target, pollicy model등 알고리즘에 맞게 추가하여 오버라이드"""
        self.model.to(self.device)
    
    def save_configs(self):
        attrs = {
            k: v.__class__.__name__ if (k in FUNC_KEYS and v is not None) else v
            for k, v in self.__dict__.items()
            if k in CONFIG_KEYS
        }
        attrs["class_name"] = self.__class__.__name__
        attrs["loss_fn"] = self.loss_fn.__name__
        attrs["optimizer_class"] = self.optimizer_class.__name__
        
        return attrs
    
    def get_attrs(self):
        """
        Config에 저장할 내용
        """
        # 데이터, 모델에 대한 config 내용 모두 trianer로 와서 config 파일에 저장하기
        attrs = {k: v.__name__ if (k in FUNC_KEYS and v is not None) else v for k, v in self.__dict__.items() if k in CONFIG_KEYS}
        attrs["class_name"] = self.__class__.__name__
        return attrs

    def get_n_success_fail(self, reward):
        """
        batch에서 성공과 실패 개수와 idx 반환
        """
        success_idx = reward.squeeze() == self.success_reward
        fail_idx = reward.squeeze() == self.fail_reward
        n_success = sum(success_idx)
        n_fail = sum(fail_idx)
            
        return n_success, n_fail, success_idx, fail_idx

    def save_checkpoint(self, state: dict, is_best: bool, filename='checkpoint', ext='.pth'):
        
        ckpt_dir = self.save_dirs["CKPT_DIRNAME"]
        pattern = f"epoch{self.current_epoch}_{self.test_metric}{self.best_metric:.2f}"
            
        filepath = os.path.join(ckpt_dir, f"{filename}_{pattern}" + ext)
        torch.save(state, filepath)
        if is_best:
            self.add_log(f"====> Save best model | {self.test_metric}:{self.best_metric}\n", "debug")
            remove_old_checkpoints(ckpt_dir, prefix="model_best", extension=ext)
            shutil.copyfile(filepath, os.path.join(ckpt_dir, f'model_best_{pattern}{ext}'))
            self.best_state = state

    @log_decorator
    def load_state(self, logger=None):
        if logger is not None:
            logger.info(f"{Fore.GREEN}==> Resuming from checkpoint..{Style.RESET_ALL}")

        self.model.load_state_dict(self.checkpoint["model_state_dict"])    
        self.start_epoch = self.checkpoint['epoch'] + 1
        self.current_epoch = self.start_epoch
        self.step = self.checkpoint["step"] + 1
    
    @log_decorator
    def load_optimizer(self, logger=None):
        if logger is not None:
            logger.info(f"{Fore.GREEN}==> load_checkpoint optimizer{Style.RESET_ALL}")
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            
        if self.scheduler is not None:
            logger.info(f"{Fore.GREEN}==> load checkpoint scheduler{Style.RESET_ALL}")
            self.scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])
    
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
            "--no_progress", action="store_true", help="don't use progress bar"
        )
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
            "--test_metric",
            type=str,
            default=TEST_METRIC,
            help="Test metric to choose the best model"
        )
        parser.add_argument(
            "--criterion",
            type=str,
            default=CRITERION,
            help="Min or Max for comparing test metric"

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
            default=OPTIMIZER,
            help="optimizer class from torch.optim",
        )
        parser.add_argument("--lr", type=float, default=LR)
        parser.add_argument(
            "--scheduler_mode",
            type=str,
            default=None,
            help="lr_schduler mode, (min, max, ...)",
        )
        parser.add_argument(
            "--loss_module",
            type=str,
            default=LOSS_MODULE_NAME,
            help="loss module name to get loss function"
        )
        parser.add_argument(
            "--loss",
            type=str,
            default=LOSS,
            help="loss function from torch.nn.functional",
        )
        return parser
