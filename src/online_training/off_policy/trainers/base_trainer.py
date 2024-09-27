import argparse
import os
import torch

from utilities.managers import (
    config_decorator, 
    logger_decorator, 
    tensorboard_decorator
)

from online_training.trainers.metrics import MetricsMeter, MetricLogs
from utilities.agent_util import is_best
from online_training.trainers.decorators import update_metrics
from utilities.util import get_config

CONFIG_IGNORE = ["args", "config"]

MIN_STEPS_BEFORE_LEARNING = 200
N_UPDATES_PER_LEARNING = 50
N_EPISODE_PER_EVAL = 200
MAX_TRAINING_STEP = 10000
N_STEP_PER_SAVING = 1000
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
    @logger_decorator
    def __init__(self, env, buffer, agent, args: argparse.Namespace = None, config=None, logger=None):
        self.config = config
        self.logger = logger
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
        self.episode_metrics = MetricsMeter() # episode 내부의 지표를 기록할 metric
        self.log_metrics = MetricLogs(rolling_window_size=self.rolling_window_size) # episode 단위로 metric 기록
        self.log_metrics_eval = MetricLogs(rolling_window_size=self.rolling_window_size) # episode 단위로 metric 기록
        self.log_metrics.add_metric_log("return", "max")
        self.log_metrics_eval.add_metric_log("return", "max")

        self.test_metric = args.get("test_metric", TEST_METRIC)
        self.criterion = args.get("test_criterion", CRITERION)

        # Training settings
        self.batch_size = args.get("batch_size", BATCH_SIZE)

        self.stage = "train"
        self.min_steps_before_learning = args.get("min_steps", MIN_STEPS_BEFORE_LEARNING)
        self.run_eval_episode = args.get("run_eval_episode", RUN_EVAL_EPISODE)
        self.max_training_steps = args.get("max_training_step", MAX_TRAINING_STEP)
        self.n_update_per_learning = args.get("n_updates_per_learning", N_UPDATES_PER_LEARNING)
        self.n_step_per_eval = args.get("n_step_per_eval", N_STEP_PER_EVAL)

        self._config = get_config(self, CONFIG_IGNORE)

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
        self.episode_metrics.reset_all()

    def track_recent_episodes(self):
        """Saves the data from the recent episodes"""
        self.states_in_episode.append(self.state)
        self.actions_in_episode.append(self.action)
        self.rewards_in_episode.append(self.reward)
        self.next_states_in_episode.append(self.next_state)
        self.dones_in_episode.append(self.done)
    
    def update_log_metrics(self):
        key_mapping = {"reward": "return"}
        for key, metric in self.episode_metrics:
            if key in key_mapping.keys():
                if self.is_eval:
                    self.log_metrics_eval[key].update(metric.sum, n_repeat_for_eval=self.n_episode_per_eval)
                else:
                    self.log_metrics[key].update(metric.sum)
            else:
                if self.is_eval:
                    self.log_metrics_eval[key].update(metric.sum, n_repeat_for_eval=self.n_episode_per_eval)
                else:
                    self.log_metrics[f"{key}/sum"].update(metric.sum)

    @update_metrics
    def run_step_in_episode(self):
        self.action = self.agent.sample_action(self.state, is_eval=self.is_eval)

        self.next_state, self.reward, terminated, truncated, info = self.env.step(self.action)
        self.done = terminated or truncated

        if self.time_to_learn:
            for _ in self.n_update_per_learning:
                self.agent.update(self.buffer.sample_batch())

        if not self.is_eval:
            self.buffer.add_experience(state=self.state, action=self.action, next_state=self.next_state, \
                                       reward=self.reward, done=self.done)

        self.state = self.next_state
        self.agent.global_step_num += 1
        self.steps_in_episode += 1

        return {"reward": self.reward}
    
    def run_episodes(self):
        """
        episode 동작
        """
        while self.agent.global_episode_num < self.max_episodes:
            self.reset_env()
            while not self.done:
                self.run_step_in_episode()
            
            self.update_log_metrics()
            self.agent.global_episode_num += 1

    @property
    def is_eval(self):
        """
        evaluation 단계 확인
        """
        return (self.steps_in_episode % self.n_episode_per_eval == 0) and self.run_eval_episode
    
    @property
    def time_to_learn(self):
        """
        buffer_size가 batch_size 보다 큰지, 최소 buffer_size, n_step learning 체크.
        """
        return self.check_enough_experiences_num_in_buffer() and self.check_minimum_step_before_training() and self.check_n_step_learning()

    def check_minimum_step_before_training(self):
        """
        학습을 하기 전 채워야할 최소 buffer 사이즈 체크
        """
        return self.agent.global_step_num > self.min_steps_before_learning
    
    def check_n_step_learning(self):
        """
        self.n_step_learning 에 한번씩 학습
        """
        return self.agent.global_episode_num % self.agent.n_step_learning == 0

    def check_enough_experiences_num_in_buffer(self):
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
            "--n_step_per_saving", type=int, default=N_STEP_PER_SAVING, help="number of step per saving checkpoint"
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
            '--n_episode_per_eval',
            type=int,
            default=N_EPISODE_PER_EVAL,
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
    
    def save_checkpoint(self, state_dict, filename="ckpt", ext=".pth"):
        ckpt_dir = self.checkpoint["dirs"]["ckpt"]
        current_score = self.log_metrics[self.test_metric].moving_avg_log[-1]
        best_score = self.log_metrics[self.test_metric].best_moving_avg
        pattern = f"e{self.agent.global_episode_num}s{self.agent.global_step_num}m{self.test_metric}b{current_score:.2f}"
        
        if is_best(best_score, current_score) or ((self.agent.global_step_num + 1) % self.n_episode_ == 0):
            filepath = os.path.join(ckpt_dir, f"{filename}_{pattern}" + ext)
            torch.save(state_dict, filepath)

            self.add_log(f"====> Save best model | {self.test_metric}:{self.best_metric}\n", "debug")
            remove_old_checkpoints(ckpt_dir, prefix="model_best", extension=ext)
            shutil.copyfile(filepath, os.path.join(ckpt_dir, f'model_best_{pattern}{ext}'))
            self.best_state = state


def save_checkpoint(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        state_dict = func(self, *args, **kwargs)
        if is_best(self.best_score, self.metric) or ((self.current_epoch + 1) % self.epochs_per_save == 0):


            self.agent.save_checkpoint(state, is_best=is_best)
            
        if is_best:
            self.add_log(f"Best metric\n {self.stage}: {outputs}")
        
        return outputs
    return wrapper
    