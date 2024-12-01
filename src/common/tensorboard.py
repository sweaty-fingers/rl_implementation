"""
Config, Logger, Tensorboard 와 같은 한 프로젝트 전반에 걸쳐 한번만 정의돼서 공유되는 클래스들 정의.
  - python metaclass를 이용한 singleton pattern 이용.
"""

# singleton_logger.py
import os
import sys
import json
import logging
import shutil

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from functools import wraps
from colorama import Fore, Style, init
from utilities.util import save_dict_to_json, get_attr_from_module, change_file_extension


# colorama 초기화
init(autoreset=True)

class ColoredFormatter(logging.Formatter):
    """
    logger에 색상 포맷 추가
    """
    def format(self, record):
        log_message = super().format(record)
        return log_message

class SingletonMeta(type):
    """
    Singleton 생성을 위한 metaclass
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
            
        return cls._instances[cls]
        
# singleton_tensorboard
class TensorBoardManager(metaclass=SingletonMeta):
    """
    단일 tensorboard 공유를 위한 singleton 클래스
    """
    def __init__(self, log_dir='/mnt/data'):
        self._writer = SummaryWriter(log_dir=log_dir)
    
    @property
    def writer(self):
        return self._writer

def tensorboard_decorator(func):
    """
    함수에 전달된 logger가 None일 경우 singleton logger 전달.
    None이 아닐 경우 전달된 logger 사용
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs["tb_writer"] is None:
            if TensorBoardManager._instances.get(TensorBoardManager):
                TensorBoardManager()
            writer_instance = TensorBoardManager._instances.get(TensorBoardManager)
            if writer_instance:
                writer = writer_instance.get_writer()
            else:
                writer = None
            kwargs["tb_writer"] = writer
        return func(*args, **kwargs)
    return wrapper

def update_tensorboard(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        outputs = func(self, *args, **kwargs)
        
        if TensorBoardManager._instances.get(TensorBoardManager):
            TensorBoardManager()
        writer_instance = TensorBoardManager._instances.get(TensorBoardManager)
        
        if writer_instance:
            writer = writer_instance.get_writer()
            for key, value in outputs.items():
                if isinstance(value, (int, float, np.integer, np.floating, torch.Tensor)):
                    writer.add_scalar(f"{self.stage}/{key}", value, self.current_epoch)
                else:
                    print(f"Unsupported data type: {type(value)} to add tensorboard scaler")
        return outputs
    return wrapper


if __name__ == "__main__":
    tb_1 = TensorBoardManager("./run1")
    tb_2 = TensorBoardManager("./run2")
    print(f"{tb_1.get_writer().log_dir}, {tb_2.get_writer().log_dir}")
    
    shutil.rmtree("./run1")
