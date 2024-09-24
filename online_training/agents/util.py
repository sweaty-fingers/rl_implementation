import os
import pandas as pd
import torch.nn as nn


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    """
    target network 업데이트.

    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def get_best_metric(best, compare, criterion: str):
    """
    현재까지의 best metric (best) 와 비교 값(compare)을 비교하여 best 값 반환
    criterion: best metric의 기준 (min or max) 
    """
    possible_criterion = ["min", "max"]
    if criterion not in possible_criterion:
        raise ValueError(f"criterion should be in {possible_criterion}\n Now get {criterion}")
    
    if criterion == "min":
        return min(best, compare), compare < best    
    elif criterion == "max":
        return max(best, compare), compare > best

def save_to_csv(outputs: dict, save_dir, stage="test"):
    """
    학습 중 기록된 dictionary 형태의 metric을 csv로 저장.
    """
    outputs = pd.DataFrame([outputs.values()], columns=outputs.keys())
    outputs.to_csv(os.path.join(save_dir, f"{stage}_output" + ".csv"))
    return outputs
