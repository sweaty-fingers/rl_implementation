import argparse
import numpy as np
import torch

from utilities.managers import (
    config_decorator, 
    logger_decorator, 
    tensorboard_decorator
)

from online_training.trainers.metrics import MetricsMeter, MetricLogs, get_best_metric
from online_training.trainers.decorators import update_metrics

TEST_METRIC = "return"
CRITERION = "max" 
MIN_STEPS_BEFORE_LEARNING = 200
N_UPDATES_PER_LEARNING = 50
N_EPISODE_PER_EVAL = 200
MAX_EPISODES = 10000
EPOCHS_PER_SAVE = 1000
RUN_EVAL_EPISODE = True
N_STEP_LEARNING = 1
BATCH_SIZE = 64
ROLLING_WINDOW_SIZE = 100

class BaseTrainer():
    """
    Agent와 Environment의 상호 작용, 학습을 담당하는 클래스
    """
    @config_decorator
    @logger_decorator
    def __init__(self, env, agent, buffer, args: argparse.Namespace = None, config=None, logger=None):
        self.config = config
        self.logger = logger
        self.args = vars(args) if args is not None else {}
        self.env = env
        self.agent = agent
        self.buffer = buffer

        # Set Device
        self.device = "cpu"
        self.gpus = self.args.get("gpus", None)
        if self.gpus is not None:
            if torch.cuda.is_available():
                self.device = f"cuda:{self.gpus}"
            elif torch.backends.mps.is_available():
                self.device = f"mps:{self.gpus}"

        # Checkpoints  
        self.save_dirs = self.config["dirs"]["checkpoint"]
        self.best_state = None

        # Metric setting
        self.rolling_window_size = self.args.get("rolling_window_size", ROLLING_WINDOW_SIZE)
        self.test_metric = self.args.get("test_metric", TEST_METRIC)
        self.criterion = self.args.get("test_criterion", CRITERION)

        self.episode_metrics = MetricsMeter() # episode 내부의 지표를 기록할 metric
        self.log_metrics = MetricLogs(rolling_window_size=self.rolling_window_size) # episode 단위로 metric 기록
        self.log_metrics_eval = MetricLogs(rolling_window_size=self.rolling_window_size) # episode 단위로 metric 기록
        self.log_metrics.add_metric_log("return", "max")
        self.log_metrics_eval.add_metric_log("return", "max")

        # Training settings
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.n_step_learning = self.args.get("n_step_learning", N_STEP_LEARNING) # n_step td

        self.stage = "train"
        self.global_step_num = 0 # 전체 학습 step
        self.global_episode_num = 0 # 진행된 전체 episode number
        self.steps_in_episode = 0 # episode내 step
        self.min_steps_before_learning = self.args.get("min_steps", MIN_STEPS_BEFORE_LEARNING)
        self.run_eval_episode = self.args.get("run_eval_episode", RUN_EVAL_EPISODE)
        self.max_episodes = self.args.get("max_episode", MAX_EPISODES)
        self.n_update_per_learning = self.args.get("n_updates_per_learning", N_UPDATES_PER_LEARNING)
        self.n_episode_per_eval = self.args.get("episode_num_per_eval", N_EPISODE_PER_EVAL)

        # Rescent episode infos:
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = False
        self.states_in_episode = []
        self.actions_in_episode = []
        self.next_states_in_episode = []
        self.rewards_in_episode = []
        self.dones_in_episode = []

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
        self.global_step_num += 1
        self.steps_in_episode += 1

        return {"reward": self.reward}
    
    def run_episodes(self):
        """
        episode 동작
        """
        while self.global_episode_num < self.max_episodes:
            self.reset_env()
            while not self.done:
                self.run_step_in_episode()
            
            self.update_log_metrics()
            self.global_episode_num += 1

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
        학습을 하기 전 채워야할 최소 buffer 사이즈 체
        """
        return self.global_step_num > self.min_steps_before_learning
    
    def check_n_step_learning(self):
        """
        self.n_step_learning 에 한번씩 학습크
        """
        return self.global_episode_num % self.n_step_learning == 0

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
            "--rolling_window_size",
            type=int,
            default=ROLLING_WINDOW_SIZE,
            help="Size of rolling window"
        )
        parser.add_argument(
            '--n_step_learning',
            type=int,
            default=N_STEP_LEARNING,
            help="N_steps to get td error"
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
            '--max_episodes',
            type=int,
            default=MAX_EPISODES,
            help="Max number of episodes"
        )

        return parser
    

    