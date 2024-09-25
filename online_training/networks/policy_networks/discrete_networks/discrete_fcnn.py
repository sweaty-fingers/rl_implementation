import argparse
import torch.nn as nn
import torch.nn.functional as F

from online_training.networks.util import get_mlp
from utilities.managers import config_decorator

FC_DIMS = [512, 256, 128]
CONFIG_IGNORE = ["data_config", "args"]

# 이산적 정책 네트워크
class DiscretePolicyNetwork(nn.Module):
    def __init__(self, data_config: dict, args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.state_dim = config["env"]["state_dim"]
        self.action_dim = config["env"]["action_dim"]
        self.fc_dims = self.args.get("fc_dims", FC_DIMS)

        self.fc_input = nn.Linear(self.state_dim, self.fc_dims[0])
        self.fc_hidden = get_mlp(self.fc_dims, activation=nn.ReLU, output_activation=nn.ReLU)
        self.fc_output = nn.Linear(self.fc_dims[-1], self.action_dim)
        
    def forward(self, state):
        x = self.fc_input(state)
        x = self.fc_hidden(x)
        x = self.fc_output(x)
        action_probs = F.softmax(x, dim=-1)

        return action_probs

    def get_config(self):
        json_compatible_types = (str, int, float, bool, list, dict, type(None))

        def is_json_compatible(value):
            # 중첩된 객체가 있을 경우 재귀적으로 처리
            if isinstance(value, json_compatible_types):
                return True
            if hasattr(value, "add_config"):
                return True
            return False

        def convert_value(value):
            if isinstance(value, json_compatible_types):
                return value
            if hasattr(value, "add_config"):
                return value.to_json_dict()
            return None  # 변환할 수 없는 값은 None으로 처리

        return {key: convert_value(value) for key, value in self.__dict__.items() if is_json_compatible(value)}

    @staticmethod
    def add_to_argparse(parser):
        return parser