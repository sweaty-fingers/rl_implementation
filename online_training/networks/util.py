"""
Network를 구성하는데 필요한 함수들 작성.
"""
import torch.nn as nn

def get_mlp(sizes, activation, output_activation=nn.Identity):
    """
    mlp(fcnn) 을 만들깅 위한 편의 함수.
    """
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)