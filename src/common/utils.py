import glob
import csv
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

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def set_seed(seed, env: Optional[gym.Env] = None):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device(device: Union[th.device, str] = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    만약 여러 gpu가 세팅된 상태에서 특정 gpu를 사용하고 싶다면, device = f"cuda:{gpu_id}" 와 같이 사용

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    
    
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = torch.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device

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

def make_config(obj, ignore):
    """
    인스턴스를 받아서 json과 호환되는 애트리뷰트를 dictionary 형식으로 반환
    """
    json_compatible_types = (str, int, float, bool, list, dict, type(None))

    def is_json_compatible(value):
        # 중첩된 객체가 있을 경우 재귀적으로 처리
        if isinstance(value, json_compatible_types):
            return True
        return False

    return {key: value for key, value in obj.__dict__.items() if is_json_compatible(value) and key not in ignore}

def save_dict_to_csv(outputs:dict, filepath):
    
    # CSV 파일이 있는지 확인
    file_exists = os.path.isfile(filepath)
    
    # CSV 파일에 데이터 추가
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=outputs.keys())
        
        # 파일이 없으면 컬럼 헤더 추가
        if not file_exists:
            writer.writeheader()
        
        # 새로운 데이터 추가
        writer.writerow(outputs)

def find_files(base_dir, pattern, file_extension):
    matched_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if pattern in file and file.endswith(file_extension):
                matched_files.append(os.path.join(root, file))
    
    return matched_files