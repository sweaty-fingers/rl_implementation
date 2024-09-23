""" Discrete action space 에서의 sac trainer """

import argparse
import torch

from colorama import Fore, Style, init
from utils.util import remove_old_checkpoints, get_attr_from_module
from utils.managers import (
    ConfigManager, 
    log_decorator, 
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

from online_training.trainers.off_policys.actor_critic.base import BaseTrainer
from online_training.trainers.metrics import MetricsMeter

OPTIMIZER_MODULE_NAME = "torch.optim"
OPTIMIZER = "Adam"
LR_POLICY = 1e-4
LR_CRITIC = 1e-4

LOSS_MODULE_NAME = "online_training.off_policys.actor_critic.losses"
LOSS = "q_loss"

CONFIG_IGNORE = []

TEST_METRIC = "average_return"
CRITERION = "max"
POSSIBLE_CRITERION = ["min", "max"]

ALPHA = 0.2
GAMMA = 0.99 

STEPS_PER_EPOCH = 2000
STEPS_PER_EPOCH_VALID = 100
EPOCHS_PER_SAVE = 1000

class DiscreteSACTrainer(BaseTrainer):
    """
    Discrete action space에서의 soft actor-critic trainer
    """
    def __init__(self, actor, critic_1, critic_2, args: argparse.Namespace = None):
        super().__init__(args=args)
        self.args = vars(args) if args is not None else {}

        # Models
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        
        # Optimizers
        self.actor_optimizer, self.critic_1_optimizer, self.critic_2_optimizer = None, None, None
        optimizer_module = self.args.get("optimizer_module", OPTIMIZER_MODULE_NAME)
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = get_attr_from_module(optimizer_module, optimizer)

        self.lr_policy = self.args.get("lr_policy", LR_POLICY)
        self.lr_critic = self.args.get("lr_critic", LR_CRITIC)
        
        # loss
        loss_module = self.args.get("loss_module", LOSS_MODULE_NAME)
        loss = self.args.get("loss", LOSS)
        self.loss_fn = get_attr_from_module(loss_module, loss)

        self.add_log(f"{Fore.GREEN}Loss fuction: {self.loss_fn.__name__}{Style.RESET_ALL}", level="info")
        
        self.alpha = self.args.get("alpha", ALPHA)
        self.gamma = self.args.get("gamma", GAMMA)

    def configure_optimizers(self):
        """
        actor와 critic의 optimizer 정의
        """
        self.actor_optimizer = self.optimizer_class(self.actor.parameters(), lr=self.lr_policy)
        self.critic_1_optimizer = self.optimizer_class(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_optimizer = self.optimizer_class(self.critic_2.parameters(), lr=self.lr_critic)

        self.add_log(f"{Fore.GREEN}==> Configure optimizer{Style.RESET_ALL}", "info")
        self.add_log(f"{Fore.GREEN}Actor optimizer: {self.actor_optimizer.__class__}{Style.RESET_ALL}", "info")
        self.add_log(f"{Fore.GREEN}Critic_1 optimizer: {self.critic_1_optimizer.__class__}{Style.RESET_ALL}", "info")
        self.add_log(f"{Fore.GREEN}Critic_2 optimizer: {self.critic_2_optimizer.__class__}{Style.RESET_ALL}", "info")

        return {"optimizer/actor": self.actor_optimizer, "optimizer/critic_1": self.critic_1_optimizer, "optimizer/critic_2": self.critic_2_optimizer}

    # 행동을 샘플링
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state)
        action = action_probs.multinomial(num_samples=1).item()
        return action
    
    # SAC 손실 함수
    def compute_loss(self, state, action, reward, next_state, done):
        # Q 네트워크 업데이트
        q_value1 = self.q_net1(state).gather(1, action)
        q_value2 = self.q_net2(state).gather(1, action)
        
        with torch.no_grad():
            next_action_probs = self.policy_net(next_state)
            next_q_values1 = self.q_net1(next_state)
            next_q_values2 = self.q_net2(next_state)
            next_q_values = torch.min(next_q_values1, next_q_values2)
            
            next_value = (next_action_probs * (next_q_values - self.alpha * torch.log(next_action_probs))).sum(dim=1, keepdim=True)
            target_q_value = reward + (1 - done) * self.gamma * next_value
        
        q_loss1 = F.mse_loss(q_value1, target_q_value)
        q_loss2 = F.mse_loss(q_value2, target_q_value)
        
        # 정책 네트워크 업데이트
        action_probs = self.policy_net(state)
        q_values1 = self.q_net1(state)
        q_values2 = self.q_net2(state)
        q_values = torch.min(q_values1, q_values2)
        
        policy_loss = (action_probs * (self.alpha * torch.log(action_probs) - q_values)).sum(dim=1).mean()
        
        return q_loss1, q_loss2, policy_loss

    # 학습 단계
    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # 손실 계산
        q_loss1, q_loss2, policy_loss = self.compute_loss(state, action, reward, next_state, done)
        
        # Q 네트워크 업데이트
        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        self.q_optimizer1.step()
        
        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        self.q_optimizer2.step()
        
        # 정책 네트워크 업데이트
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()