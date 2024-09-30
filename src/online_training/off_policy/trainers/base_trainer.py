import argparse
import csv
import os
import shutil
import torch

from utilities.managers import (
    config_decorator, 
)

from utilities.metrics import MetricsMeter, LogMetrics
from utilities.trainer_util import is_best
from utilities.util import make_config, remove_old_checkpoints, save_dict_to_csv
from utilities.logger_util import add_log

CONFIG_IGNORE = ["args", "config"]

MIN_STEPS_BEFORE_LEARNING = 200
N_UPDATES_PER_LEARNING = 50
EVAL_EVERY_N_STEPS = 200
SAVE_EVERY_N_STEPS = 200
UPDATE_EVERY_N_STEPS = 200
MAX_TRAINING_STEP = 10000
SAVING = 1000
RUN_EVAL_EPISODE = True
BATCH_SIZE = 64
ROLLING_WINDOW_SIZE = 100

TEST_METRIC = "return"
CRITERION = "max"

class BaseTrainer():
    """
    Agent와 Environment의 상호 작용, 학습을 담당하는 클래스
    """
    @config_decorator
    def __init__(self, env, buffer, agent, args: argparse.Namespace = None, config=None):
        self.config = config
        args = vars(args) if args is not None else {}
        self.env = env
        self.buffer = buffer
        self.agent = agent

        # Set Device
        self.device = "cpu"
        self.gpus = args.get("gpus", None)
        if self.gpus is not None:
            if torch.cuda.is_available():
                self.device = f"cuda:{self.gpus}"
            elif torch.backends.mps.is_available():
                self.device = f"mps:{self.gpus}"

        # Checkpoints  
        self.checkpoint = self.config["checkpoint"]
        
        # Metric setting
        self.rolling_window_size = args.get("rolling_window_size", ROLLING_WINDOW_SIZE)
        self.metrics = MetricsMeter() # 단일 episode 지표를 기록할 metric
        self.log_metrics = LogMetrics(rolling_window_size=self.rolling_window_size) # episode 단위로 metric 기록
        self.log_metrics_eval = LogMetrics(rolling_window_size=self.rolling_window_size) # episode 단위로 metric 기록
        self.log_metrics.add_metric_log("return", "max")
        self.log_metrics_eval.add_metric_log("return", "max")

        self.test_metric = args.get("test_metric", TEST_METRIC)
        self.criterion = args.get("test_criterion", CRITERION)

        # Training settings
        self.batch_size = args.get("batch_size", BATCH_SIZE)

        self.stage = "train"
        self.min_steps_before_learning = args.get("min_steps", MIN_STEPS_BEFORE_LEARNING)
        self.run_eval_episode = args.get("run_eval_episode", RUN_EVAL_EPISODE)
        self.max_training_step = args.get("max_training_step", MAX_TRAINING_STEP)
        self.n_updates_per_learning = args.get("n_updates_per_learning", N_UPDATES_PER_LEARNING)
        self.update_every_n_steps = args.get("update_every_n_steps", UPDATE_EVERY_N_STEPS)
        self.eval_every_n_steps = args.get("eval_every_n_steps", EVAL_EVERY_N_STEPS)
        self.save_every_n_steps = args.get("save_every_n_steps", SAVE_EVERY_N_STEPS)

        self._config = make_config(self, CONFIG_IGNORE)

        # Rescent episode infos:
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = False
        self.steps_in_episode = 0 # episode내 step
        self.states_in_episode = []
        self.actions_in_episode = []
        self.next_states_in_episode = []
        self.rewards_in_episode = []
        self.dones_in_episode = []

    @property
    def config(self):
        return self._config

    def reset_env(self):
        """
        에피소드 종료 후 환경 리
        """
        self.state = self.env.reset() # env의 첫 번째 요소 state 반환
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = False
        self.steps_in_episode = 0
        self.states_in_episode = []
        self.actions_in_episode = []
        self.next_states_in_episode = []
        self.rewards_in_episode = []
        self.dones_in_episode = []

        # Metric
        self.metrics.reset_all()
    
    def run_episodes(self):
        """
        episode 동작
        """
        while self.agent.global_step_num < self.max_training_step:
            self.reset_env()
            while not self.done:
                outputs = self.run_step_in_episode()
                self.update_metrics(outputs=outputs)
                self.track_recent_episode()
            
            self.update_log_metrics()
            self.agent.global_episode_num += 1

    def run_step_in_episode(self):
        self.action = self.agent.sample_action(self.state, is_eval=self.is_eval)

        self.next_state, self.reward, terminated, truncated, info = self.env.step(self.action)
        self.done = terminated or truncated

        if self.time_to_learn:
            for _ in self.n_updates_per_learning:
                self.agent.update(self.buffer.sample_batch())

        if not self.is_eval:
            self.buffer.add_experience(state=self.state, action=self.action, next_state=self.next_state, \
                                       reward=self.reward, done=self.done)

        self.state = self.next_state
        self.agent.global_step_num += 1
        self.steps_in_episode += 1

        return {"reward": self.reward}

    def update_log_metrics(self):
        key_mapping = {"reward": "return"}

        if self.is_eval:
            log_metrics = self.log_metrics_eval
        else:
            log_metrics = self.log_metrics

        for key, metric in self.metrics.items():
            if key in key_mapping:
                log_metrics[key_mapping[key]].update(metric.sum, n_repeat_for_eval=self.eval_every_n_steps)
            else:
                log_metrics[key_mapping[key]].update(metric.sum, n_repeat_for_eval=self.eval_every_n_steps)

    def update_metrics(self, outputs):
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].update(value)

    def save_metrics_csvs(self):
        csv_path = os.path.join(self.checkpoint["dirs"]["csv"], "metrics")
        if self.is_eval:
            metrics = self.log_metrics_eval
            filepath = os.path.join(csv_path, "metrics_evl.csv")
        else:
            metrics = self.log_metrics_eval
            filepath = os.path.join(csv_path, "metrics.csv")

        save_dict_to_csv(metrics.recent_values_to_dict(), filepath=filepath)

    def track_recent_episode(self):
        """Saves the data from the recent episodes"""
        self.states_in_episode.append(self.state)
        self.actions_in_episode.append(self.action)
        self.rewards_in_episode.append(self.reward)
        self.next_states_in_episode.append(self.next_state)
        self.dones_in_episode.append(self.done)
    
    def save_checkpoint(self, state_dict, filename="ckpt", ext=".pth"):
        ckpt_dir = self.checkpoint["dirs"]["ckpt"]
        current_score = self.log_metrics[self.test_metric].moving_avg_log[-1]
        best_score = self.log_metrics[self.test_metric].best_moving_avg
        
        if is_best(best_score, current_score) or ((self.agent.global_step_num) % self.save_every_n_steps == 0):
            pattern = f"e{self.agent.global_episode_num}s{self.agent.global_step_num}m{self.test_metric}b{current_score:.2f}"
            filepath = os.path.join(ckpt_dir, f"{filename}_{pattern}" + ext)
            torch.save(state_dict, filepath)

            # best_state는 model_best로 시작하는 checkpoint로 저장.
            add_log(f"====> Save best model | {self.test_metric}:{best_score}\n", "debug")
            remove_old_checkpoints(ckpt_dir, prefix="model_best", extension=ext)
            shutil.copyfile(filepath, os.path.join(ckpt_dir, f'model_best_{pattern}{ext}'))

    @property
    def is_eval(self):
        """
        evaluation 단계 확인
        """
        return (self.steps_in_episode % self.eval_every_n_steps == 0) and self.run_eval_episode
    
    @property
    def time_to_learn(self):
        """
        buffer_size가 batch_size 보다 큰지, 최소 buffer_size, n_step learning 체크.
        """
        return self.check_enough_experiences_in_buffer and self.check_minimum_step_before_training and self.check_update_every_n_steps
    
    @property
    def check_minimum_step_before_training(self):
        """
        학습을 하기 전 채워야할 최소 buffer 사이즈 체크
        """
        return self.agent.global_step_num > self.min_steps_before_learning
    
    @property
    def check_update_every_n_steps(self):
        """
        self.n_step_learning 에 한번씩 학습
        """
        return self.agent.global_step_num % self.update_every_n_steps == 0

    @property
    def check_enough_experiences_in_buffer(self):
        """
        버퍼 안에 충분한 수의 experience(>batch_size)가 있는지 확인
        """
        return len(self.buffer) > self.batch_size
        
    @staticmethod
    def add_to_argparse(parser):
        """
        Trainer에 필요한 parser 정의
        """
        parser.add_argument(
            "--no_progress", action="store_true", help="don't use progress bar"
        )
        parser.add_argument(
            "--run_eval_episode", action="store_true", help="Run eval_episode during training"
        )
        parser.add_argument(
            "--gpus",
            default=None,
            type=int,
            help="id(s) for GPU_VISIBLE_DEVICES(MPS or CUDA)",
        )
        parser.add_argument(
            "--save_every_n_steps", type=int, default=SAVE_EVERY_N_STEPS, help="number of step per saving checkpoint"
        )
        parser.add_argument(
            "--rolling_window_size",
            type=int,
            default=ROLLING_WINDOW_SIZE,
            help="Size of rolling window"
        )
        parser.add_argument(
            '--min_steps_before_training',
            type=int,
            default=MIN_STEPS_BEFORE_LEARNING,
            help="Minimum steps before learning"
        )
        parser.add_argument(
            '--n_updates_per_learning',
            type=int,
            default=N_UPDATES_PER_LEARNING,
            help="Number of updates for learning"
        )
        parser.add_argument(
            '--update_every_n_steps',
            type=int,
            default=UPDATE_EVERY_N_STEPS,
            help = "Update agent every n steps"
        )
        parser.add_argument(
            '--eval_every_n_steps',
            type=int,
            default=EVAL_EVERY_N_STEPS,
            help="Number of episode per running eval"
        )
        parser.add_argument(
            '--max_training_step',
            type=int,
            default=MAX_TRAINING_STEP,
            help="Max number of steps for training"
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

        return parser