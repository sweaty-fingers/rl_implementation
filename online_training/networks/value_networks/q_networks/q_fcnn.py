import argparse
import torch.nn as nn
import torch.nn.functional as F

from online_training.networks.util import get_mlp

FC_DIMS = [512, 256, 128]
CONFIG_IGNORE = ["data_config"]

# 이산적 정책 네트워크
class FCNNQNetwork(nn.Module):
    def __init__(self, data_config, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.state_dim = data_config["state_dim"]
        self.action_dim = data_config["action_dim"]

        self.fc_dims = self.args.get("fc_dims", FC_DIMS)

        self.fc_input = nn.Linear(self.state_dim, self.fc_dims[0])
        self.fc_hidden = get_mlp(self.fc_dims, activation=nn.ReLU, output_activation=nn.ReLU)
        self.fc_output = nn.Linear(self.fc_dims[-1], self.action_dim)
        
    def forward(self, state):
        x = self.fc_input(state)
        x = self.fc_hidden(x)
        q_values = self.fc_output(x)
        return q_values

    @staticmethod
    def add_to_argparse(parser):
        return parser