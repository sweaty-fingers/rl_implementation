import argparse
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional
from online_training.networks.util import get_mlp
from utilities.util import get_config

CONFIG_IGNORE = ["args"]
FC_DIMS = [512, 256, 128]

# 이산적 정책 네트워크
class FCNNQNetwork(nn.Module):
    def __init__(self, config, fc_dims: Optional[list] = None, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.state_dim = config["env"]["state_dim"]
        self.action_dim = config["env"]["action_dim"]

        self.fc_dims = fc_dims
        if self.fc_dims is None:
            self.fc_dims = self.args.get("fc_dims", FC_DIMS)

        self._config = get_config(self, CONFIG_IGNORE)

        self.fc_input = nn.Linear(self.state_dim, self.fc_dims[0])
        self.fc_hidden = get_mlp(self.fc_dims, activation=nn.ReLU, output_activation=nn.ReLU)
        self.fc_output = nn.Linear(self.fc_dims[-1], self.action_dim)

    @property
    def config(self):
        return self._config
        
    def forward(self, state):
        x = self.fc_input(state)
        x = self.fc_hidden(x)
        q_values = self.fc_output(x)
        return q_values

    @staticmethod
    def add_to_argparse(parser):
        """
        cli로 받을 수 있는 설정 추가
        """
        parser.add_argument("--fc_dims", nargs="+", default=FC_DIMS)
        # '+': 한개 이상의 값 
        # 받는 예시: python script.py --fc_dims 1 2 3 4 5
        return parser