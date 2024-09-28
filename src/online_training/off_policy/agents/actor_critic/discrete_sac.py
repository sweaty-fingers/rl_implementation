""" Discrete action space 에서의 sac trainer """

import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from colorama import Fore, Style
from utilities.util import get_attr_from_module, get_config
from utilities.torch_util import create_actor_distribution
from utilities.managers import tensorboard_decorator
from utilities.logger_util import add_log
from training.setup import import_class

from online_training.agents.off_policys.actor_critic.base import BaseAgent
from online_training.agents.util import soft_update


CONFIG_IGNORE = ["args"]

OPTIMIZER_MODULE_NAME = "torch.optim"
OPTIMIZER = "Adam"
LR_POLICY = 1e-4
LR_CRITIC = 1e-4

ALPHA = 0.2
GAMMA = 0.99
CRITIC_TAU = 0.005

AUTOMATIC_ENTROPY_TUNING = False # entropy를 학습 가능한 파라미터로 설정

class DiscreteSACAgent(BaseAgent):
    """
    Discrete action space에서의 soft actor-critic trainer
    """
    def __init__(self, actor, critic_1, critic_2, args: argparse.Namespace = None, config=None, logger=None):
        super().__init__(args=args, config=config, logger=logger)
        self.args = vars(args) if args is not None else {}
        # Models
        self.actor = actor
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.target_critic_1, self.target_critic_2 = deepcopy(self.critic_1), deepcopy(self.critic_2)
        self.critic_tau = self.args.get("target_critic_tau", CRITIC_TAU)

        self.add_network_graph_in_tb_writer()

        # Optimizers
        self.actor_optimizer, self.critic_1_optimizer, self.critic_2_optimizer = None, None, None
        optimizer_module = self.args.get("optimizer_module", OPTIMIZER_MODULE_NAME)
        optimizer = self.args.get("optimizer", OPTIMIZER)
        self.optimizer_class = get_attr_from_module(optimizer_module, optimizer)

        self.lr_policy = self.args.get("lr_policy", LR_POLICY)
        self.lr_critic = self.args.get("lr_critic", LR_CRITIC)
        self.configure_optimizers()

        self.automatic_entropy_tuning = self.args.get("automatic_entropy_tuning", AUTOMATIC_ENTROPY_TUNING)
        if self.automatic_entropy_tuning:
            # we set the max possible entropy as the target entropy
            # entropy는 max_entropy (모든 액션의 확률이 동일)로 고정. alpha값을 학습 가능한 파라미터로 설정.
            self.target_entropy = -np.log((1.0 / self.action_dim)) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr_policy, eps=1e-4)
            add_log(f"{Fore.GREEN}Set automatic entropy tunning{Style.RESET_ALL}", level="info")
            add_log(f"{Fore.GREEN}Trainable variable alpha: {self.alpha}{Style.RESET_ALL}", level="info")
        else:
            self.alpha = self.args.get("alpha", ALPHA)
            self.alpha_optimizer = None
            add_log(f"{Fore.GREEN}Not automatic entropy tunning{Style.RESET_ALL}", level="info")
            add_log(f"{Fore.GREEN}Constant alpha: {self.alpha}{Style.RESET_ALL}", level="info")
        
        self._config = get_config(self, CONFIG_IGNORE)

    @property
    def config(self):
        return self._config

    @tensorboard_decorator
    def add_network_graph_in_tb_writer(self, tb_writer):
        # Add model graphs in tensorboard
        if tb_writer is not None:
            add_log(f"{Fore.GREEN}Add Network graphs in tensorboard{Style.RESET_ALL}", level="info")
            dummy = torch.randn(1, self.state_dim) # batch(1), state_dim
            tb_writer.add_graph(self.actor, dummy)
            tb_writer.add_graph(self.critic_1, dummy)
            tb_writer.add_graph(self.critic_2, dummy)

    def configure_optimizers(self):
        """
        actor와 critic의 optimizer 정의
        """
        self.actor_optimizer = self.optimizer_class(self.actor.parameters(), lr=self.lr_policy)
        self.critic_1_optimizer = self.optimizer_class(self.critic_1.parameters(), lr=self.lr_critic)
        self.critic_2_optimizer = self.optimizer_class(self.critic_2.parameters(), lr=self.lr_critic)

        add_log(f"{Fore.GREEN}==> Configure optimizer{Style.RESET_ALL}", "info")
        add_log(f"{Fore.GREEN}Actor optimizer: {self.actor_optimizer.__class__}{Style.RESET_ALL}", "info")
        add_log(f"{Fore.GREEN}Critic_1 optimizer: {self.critic_1_optimizer.__class__}{Style.RESET_ALL}", "info")
        add_log(f"{Fore.GREEN}Critic_2 optimizer: {self.critic_2_optimizer.__class__}{Style.RESET_ALL}", "info")

        return {"optimizer/actor": self.actor_optimizer, "optimizer/critic_1": self.critic_1_optimizer, "optimizer/critic_2": self.critic_2_optimizer}

    def get_action_and_action_info(self, state: torch.Tensor):
        """
        state를 받아서 action과 action과 action 관련 정보 반환.

        Args:
            state (torch.Tensor): 상태 정보
        
        Output
            action (_type_): 수행할 action, 각 액션을 수행활 확률 기반하여 샘플링된 action (stochastic, training에 사용)
            action_probs: 각 액션을 수행할 확률
            log_action_probs: log(action_probs)
            max_prob_action: max_probability를 갖는 action (deterministic, evaluation에 사용)
        """
        action_probabilities = self.actor(state)
        max_probability_action = torch.argmax(action_probabilities, dim=-1)
        action_distribution = create_actor_distribution(self.action_type, action_probabilities, self.action_dim)
        action = action_distribution.sample().cpu() # action은 학습 critic loss에서 인덱싱으로만 사용.
        # Have to deal with situation of 0.0 probabilities because we can't do log 0
        z = action_probabilities == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probabilities + z)
        
        return action, (action_probabilities, log_action_probabilities), max_probability_action
    
    def sample_action(self, state, is_eval=False):
        if is_eval:
            with torch.no_grad():
                *_, action = self.get_action_and_action_info(state)
        else:
            if self.global_step_number < self.min_buffer_size:
                action = self.env.action_space.sample() # 랜덤 샘플링
                # print buffer size, action random sampling 중인 것 명시하기
            else: 
                action, *_ = self.get_action_and_action_info(state)
        
        action = action.detach().cpu().numpy()
        return action[0]

    def critic_loss(self,
                    states: torch.Tensor, 
                    actions: torch.Tensor, 
                    rewards: torch.Tensor, 
                    next_states: torch.Tensor, 
                    dones: torch.Tensor):
        """
        diecrete action space에서의 critic_loss (q_loss):
        $$
        Q^{\pi}(s,a) &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') + \alpha H\left(\pi(\cdot|s')\right) \right)} \\
        &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') - \alpha \log \pi(a'|s') \right)}
        $$
        """
        with torch.no_grad():
            # target network는 gradient에 포함 x
            next_state_action, (action_probabilities, log_action_probabilities), _ = self.get_action_and_action_info(next_states)
            qf1_next_target = self.target_critic_1(next_states)
            qf2_next_target = self.target_critic_2(next_states)
            min_qf_next_target = action_probabilities * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities) # q + H(pi)
            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = rewards + (1.0 - dones) * self.gamma * (min_qf_next_target)
        
        qf1 = self.critic_1(states).gather(1, actions.long())
        qf2 = self.critic_2(states).gather(1, actions.long())
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        
        return qf1_loss, qf2_loss
    
    def policy_loss(
        self, 
        states: torch.Tensor, 
        ):
        """
        Discrete action space에서의 policy loss
        $$
        V^{\pi}(s) &= \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right) \\
        &= \underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.
        $$
        """
        action, (action_probabilities, log_action_probabilities), _ = self.get_action_and_action_info(states)
        qf1_pi = self.critic_1(states)
        qf2_pi = self.critic_2(states)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term = self.alpha * log_action_probabilities - min_qf_pi
        policy_loss = (action_probabilities * inside_term).sum(dim=1).mean() # discrete 액션 space 이므로 action_probablities를 곱하고 더함으로써 기대값이 됨.
        entropy = torch.sum(log_action_probabilities * action_probabilities, dim=1) # 이산확률변수에서의 엔트로피
        return policy_loss, entropy
    
    def entropy_tuning_loss(self, entropy):
        """
        (??) Entropy tuning에 대해서 알아보기
        """
        alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
        return alpha_loss

    def update(self, batch):
        states, actions, rewards, next_states, dones = batch["states"], batch["actions"], batch["rewards"], batch["next_states"], batch["dones"]

        # Update critic networks
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        qf1_loss, qf2_loss = self.critic_loss(states=states, actions=actions, rewards=rewards, next_states=next_states, dones=dones)
        qf1_loss.backward()
        self.critic_1_optimizer.step()
        qf2_loss.backward()
        self.critic_1_optimizer.step()
        
        ## Update target critic networks
        self.update_target_critics()

        # Update actor network
        self.actor_optimizer.zero_grad()
        policy_loss, entropy = self.policy_loss(states=states)
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss = self.entropy_tuning_loss(entropy=entropy)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()

    def update_target_critics(self):
        soft_update(self.target_critic_1, self.critic_1, self.critic_tau)
        soft_update(self.target_critic_2, self.critic_2, self.critic_tau)

    @property
    def state_dict(self) -> dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic_1.state_dict(),
            "critic2": self.critic_2.state_dict(),
            "critic1_target": self.target_critic_1.state_dict(),
            "critic2_target": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_1_optimizer": self.critic_1_optimizer.state_dict(),
            "critic_2_optimizer": self.critic_2_optimizer.state_dict(),
            "sac_log_alpha": self.log_alpha,
            "sac_log_alpha_optimizer": self.alpha_optimizer.state_dict(),
            "global_step_number": self.global_step_num,
            "global_episode_number": self.global_episode_num
        }

    def load_state_dict(self, ckpt=None):
        if ckpt is not None:
            super().load_state_dict()
            self.actor.load_state_dict(ckpt["state_dict/actor"])
            self.critic_1.load_state_dict(ckpt["state_dict/critic_1"])
            self.critic_2.load_state_dict(ckpt["state_dict/critic_2"])
            self.actor_optimizer.load_state_dict(ckpt["state_dict/actor_optimizer"])
            self.critic_1_optimizer.load_state_dict(ckpt["state_dict/critic_1_optimizer"])
            self.critic_2_optimizer.load_state_dict(ckpt["state_dict/critic_2_optimizer"])
            if self.automatic_entropy_tuning:
                self.log_alpha.load_state_dict(ckpt["state_dict/alpha"])
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(ckpt["state_dict/alpha_optimizer"])
    
    @staticmethod
    def setup_networks(config: dict, args: argparse.Namespace = None):
        actor_network_class = import_class(f"networks.policy.{config["networks"]["actor"]["class_name"]}")
        actor = actor_network_class(config=config, args=args)

        critic_1_network_class = import_class(f"networks.value.{config["networks"]["critic_1"]["class_name"]}")
        critic_1 = critic_1_network_class(config=config, args=args)

        critic_2_network_class = import_class(f"networks.value.{config["networks"]["critic_2"]["class_name"]}")
        critic_2 = critic_2_network_class(config=config, args=args)

        return {"actor": actor, "critic_1": critic_1, "critic_2": critic_2}
        
    @staticmethod
    def add_to_argparse(parser):
        parser = BaseAgent.add_to_argparse(parser)
        parser.add_argument("--alpha", type=float, default=ALPHA)
        parser.add_argument("--target_critic_tau", type=float, default=CRITIC_TAU, help="Coefficient moving avg for updating target critic network")
        parser.add_argument("--automatic_entropy_tuning", action='store_true', help="Set alpha learnable parameter")
        
        parser.set_defaults(optimizer_module=OPTIMIZER_MODULE_NAME)
        parser.set_defaults(optimizer=OPTIMIZER)
        parser.set_defaults(lr_policy=LR_POLICY)
        parser.set_defaults(lr_critic=LR_CRITIC)
        parser.set_defaults(gamma=GAMMA)
            
        return parser
    
