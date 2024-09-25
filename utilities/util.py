import glob
import importlib
import os
import random
import shutil
import json

import numpy as np
import torch
import gymnasium as gym

from typing import Optional
from colorama import Fore, Style

def set_seed(seed, env: Optional[gym.Env] = None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_directory(path):
    if os.path.exists(path):
        response = input(f"{Fore.RED}The directory '{path}' already exists. Do you want to delete it and create a new one? (y/n): {Style.RESET_ALL}")
        if response.lower() == 'y':
            shutil.rmtree(path)
            os.makedirs(path)
            print(f"The directory '{path}' has been deleted and recreated.")
        else:
            print("Operation cancelled. The directory was not modified.")
    else:
        os.makedirs(path)
        print(f"The directory '{path}' has been created.")

def remove_old_checkpoints(directory_path, prefix="model_best", extension=".pth"):
    # 지정된 디렉토리 내에서 prefix로 시작하고 extension으로 끝나는 파일 찾기
    pattern = os.path.join(directory_path, f"{prefix}*{extension}")
    files_to_remove = glob.glob(pattern)

    # 파일 삭제
    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed old checkpoint file: {file_path}")
        except FileNotFoundError:
            print(f"File not found, skipping removal: {file_path}")
        except Exception as e:
            print(f"Error removing file {file_path}: {e}")
            raise

def save_dict_to_json(data, filepath, readonly=False):
    # 파일이 이미 존재하는지 확인
    if os.path.exists(filepath):
        # 사용자에게 덮어쓸지 여부를 묻기
        overwrite = input(f"File '{filepath}' already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("File not overwritten.")
            return
        
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)

    if readonly:
        os.chmod(filepath, 0o444)
    print(f"File '{filepath}' has been saved. Readonly:{readonly}\n")

def get_attr_from_module(module_name, attr_name):
    """
    모듈에서 attr를 반환
    """
    # 모듈 동적 임포트
    module = importlib.import_module(module_name)
    # 모듈에서 변수 가져오기
    variable = getattr(module, attr_name)
    return variable

def change_file_extension(filepath, new_extension):
    """
    경로에서 파일의 확장자 변경.
    """
    # 파일 경로와 확장자를 분리
    base, _ = os.path.splitext(filepath)
    # 새로운 확장자를 추가하여 새로운 파일 경로 생성
    new_filepath = f"{base}.{new_extension.lstrip('.')}"
    return new_filepath