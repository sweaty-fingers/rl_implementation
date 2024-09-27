import torch.nn as nn

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    target network 업데이트.

    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

