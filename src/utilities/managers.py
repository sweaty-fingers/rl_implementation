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

class ConfigManager(metaclass=SingletonMeta):
    """
    실행 과정에서 단일 config 공유를 위한 singleton 클래스
    """
    def __init__(self, config_path):
        basename, ext = os.path.splitext(config_path)
        if ext != "json":
            print(f"Change extension from {ext} to json\n")
            config_path = f"{basename}.json"

        if not os.path.exists(config_path):
            print(f"{config_path} 가 존재하지 않습니다.\n")
            py_path = change_file_extension(config_path, "py")
            if not os.path.exists(py_path):
                print(f"Config를 작성하기 위한 py 파일({py_path})이 존재하지 않습니다.")
                return
            
            print(f"{py_path} 기반으로 json 파일을 생성합니다.\n ")
            cfg = get_attr_from_module(os.path.splitext(py_path)[0].replace("/", "."), "CONFIG")
            save_dict_to_json(cfg, config_path)
            
        with open(config_path, "r") as file:
            self._config = json.load(file)

    @property
    def config(self) -> dict:
        return self._config

class LoggerManager(metaclass=SingletonMeta):
    """
    실행 과정에서 단일 logger 공유를 위한 sigleton 클래스
    """
    def __init__(self, name='my_logger', level=logging.DEBUG):
        formatter = ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s')
        
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        ## console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        self._logger.addHandler(console_handler)

        # Creating and adding the file handler
        file_debug_handler = logging.FileHandler(name + "_debug" + ".log")
        file_debug_handler.setFormatter(formatter)
        file_debug_handler.setLevel(logging.DEBUG)
    
        file_info_handler = logging.FileHandler(name + "_info" + ".log")
        file_info_handler.setFormatter(formatter)
        file_info_handler.setLevel(logging.INFO)

        self._logger.addHandler(file_debug_handler)
        self._logger.addHandler(file_info_handler)

    @property
    def logger(self):
        return self._logger
        
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

def config_decorator(func):
    """
    함수에 config 인자가 None 일 경우 singleton으로 정의된 config 전달
    None이 아닐 경우 전달된 config 사용
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("config") is None:
            if ConfigManager._instances.get(ConfigManager) is None:
                raise ValueError("You should instantiate config")
            
            config_instance = ConfigManager._instances.get(ConfigManager)
            config = config_instance.config
            kwargs["config"] = config

        return func(*args, **kwargs)
    return wrapper

def logger_decorator(func):
    """
    함수에 전달된 logger가 None일 경우 singleton logger 전달.
    None이 아닐 경우 전달된 logger 사용
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if kwargs.get("logger") is None:
            if LoggerManager._instances.get(LoggerManager) is None: # 정의되어있지 않을 경우 default logger 생성
                LoggerManager()
            
            logger_instance = LoggerManager._instances.get(LoggerManager)
            kwargs["logger"] = logger_instance.logger
            
        return func(*args, **kwargs)
    return wrapper

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


def test_configmanager():
    if len(sys.argv) < 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    
    config_manager = ConfigManager(config_path)
    cfg = config_manager.get_config()
    print(cfg)


if __name__ == "__main__":
    logger_manager_1 = LoggerManager("first_logger")
    logger_manager_2 = LoggerManager("second_logger")
    print(f"{logger_manager_1.get_logger().name} {logger_manager_2.get_logger().name}")

    tb_1 = TensorBoardManager("./run1")
    tb_2 = TensorBoardManager("./run2")
    print(f"{tb_1.get_writer().log_dir}, {tb_2.get_writer().log_dir}")
    
    shutil.rmtree("./run1")
