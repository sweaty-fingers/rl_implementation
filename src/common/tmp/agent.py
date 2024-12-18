import torch
import torch.nn as nn
from torch.distributions import Categorical, normal

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    target network 업데이트.
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def create_actor_distribution(action_types, actor_output, action_dim):
    """Creates a distribution that the actor can then use to randomly draw actions"""
    if action_types == "DISCRETE":
        assert actor_output.size()[1] == action_dim, "Actor output the wrong size"
        action_distribution = Categorical(actor_output)  # this creates a distribution to sample from
    else:
        assert actor_output.size()[1] == action_dim * 2, "Actor output the wrong size"
        means = actor_output[:, :action_dim].squeeze(0)
        stds = actor_output[:,  action_dim:].squeeze(0)
        if len(means.shape) == 2: means = means.squeeze(-1)
        if len(stds.shape) == 2: stds = stds.squeeze(-1)
        if len(stds.shape) > 1 or len(means.shape) > 1:
            raise ValueError("Wrong mean and std shapes - {} -- {}".format(stds.shape, means.shape))
        action_distribution = normal.Normal(means.squeeze(0), torch.abs(stds))
    return action_distribution