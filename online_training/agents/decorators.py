"""
decorator 함수들 정의.
"""

import os
import csv
import torch
from functools import wraps

from online_training.agents.util import get_best_metric, save_to_csv

def add_to_csv(key):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 원래 함수 호출
            outputs = func(self, *args, **kwargs)
            
            # CSV 파일 경로 동적으로 가져오기
            file_path = os.path.join(self.save_dirs["CSV_DIRNAME"], key + ".csv")
            
            # CSV 파일이 있는지 확인
            file_exists = os.path.isfile(file_path)
            
            # CSV 파일에 데이터 추가
            with open(file_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=outputs.keys())
                
                # 파일이 없으면 컬럼 헤더 추가
                if not file_exists:
                    writer.writeheader()
                
                # 새로운 데이터 추가
                writer.writerow(outputs)
            
            return outputs
        return wrapper
    return decorator

def update_scheduler(key):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            outputs = func(self, *args, **kwargs)
            if self.scheduler is not None:
                self.scheduler.step(self.metrics[key])
            return outputs
        return wrapper
    return decorator

def update_metrics(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # 원래 함수 호출
        outputs = func(self, *args, **kwargs)
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.metrics[key].update(value)
        return outputs
    return wrapper

def save_checkpoint(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        outputs = func(self, *args, **kwargs)
        self.best_metric, is_best = get_best_metric(self.best_metric, outputs[self.test_metric], criterion=self.criterion)
        
        if is_best or ((self.current_epoch + 1) % self.epochs_per_save == 0):
            state = {
                "epoch": self.current_epoch,
                "step": self.step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                self.test_metric: self.best_metric,
                # 'scheduler_state_dict': self.scheduler.state_dict(),
            }

            self.save_checkpoint(state, is_best=is_best)
            
        if is_best:
            self.add_log(f"Best metric\n {self.stage}: {outputs}")
        
        return outputs
    return wrapper

def save_result_csv(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        outputs = func(self, *args, **kwargs)
        save_to_csv(
            outputs["train"], save_dir=self.save_dirs["CSV_DIRNAME"], stage="train"
        )
        save_to_csv(
            outputs["valid"], save_dir=self.save_dirs["CSV_DIRNAME"], stage="valid"
        )
        return outputs
    return wrapper
        
def log_outputs(level):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwrags):
            outputs = func(self, *args, **kwrags)
            s = ""
            for key, value in outputs.items():
                s += f"  {key}: {value}"
            self.add_log(s, level)
            return outputs
        return wrapper
    return decorator    
