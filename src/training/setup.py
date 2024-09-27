import argparse
import glob
import importlib
import os

import gymansium_tutorials as gym
import torch

from datetime import datetime
from utilities.managers import ConfigManager, TensorBoardManager, LoggerManager
from utilities.util import create_directory

ENV_CLASS_MODULE = "envs"
SEED = 2024

def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'src.online_training.trainers.BaseTrainer'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def setup_experiment_log_dir(args: argparse.Namespace):
    """
    기존 학습 정보 존재 여부 확인,
    if) 정보(checkpoint)가 있을 경우 로드
    if) checkpoint가 없을 경우 새로운 저장 경로(experiment_log_dir) 생성

    """
    ckpt = None
    if args.checkpoint is not None:
        experiment_log_dir = args.checkpoint

        # Set config and args
        config_path = os.path.join(experiment_log_dir, "configs", "config.json")
        config = ConfigManager(config_path).config
        args = argparse.Namespace(**config["args"])

        try:
            pattern = os.path.join(config["checkpoint"]["dirs"]["ckpt"], "model_best*.pth")
            ckpt_path = glob.glob(pattern)[0]
            ckpt = torch.load(ckpt_path)

        except FileExistsError as e:
                print(f"Error: no checkpoint directory found! {e}")

        if args.max_training_step <= ckpt["global_step_num"] + 1:
            raise ValueError(f"max_training_step should be larger than current global step {ckpt['global_step_num']}")
        
    else:
        config = ConfigManager(args.config).config
        logs_save_dir = config["DIRS"]["ROOT"]["EXPERIMENT_DIRNAME"]
        experiment_description = f"{args.env}/{args.buffer}/{args.network}/{args.trainer}/{args.agent}"

        if args.max_epochs < 3:
            experiment_log_dir = os.path.join(logs_save_dir, experiment_description, "test")
        else:
            i = 1
            while True: 
                experiment_log_dir = os.path.join(logs_save_dir, experiment_description, f"run{i}")
            
                if not os.path.exists(os.path.join(experiment_log_dir, "csv")):
                    break
                i += 1
                
        checkpoint_dirs = {}
        checkpoint_dirs["root"] = experiment_log_dir
        checkpoint_dirs["config"] = os.path.join(experiment_log_dir, "configs")
        checkpoint_dirs["csv"] = os.path.join(experiment_log_dir, "csv")
        checkpoint_dirs["ckpt"] = os.path.join(experiment_log_dir, "ckpt")

        for _, d in checkpoint_dirs.items():
            create_directory(d)
        config["checkpoint"] = {}          
        config["checkpoint"]["dirs"] = checkpoint_dirs

    return experiment_log_dir, config, args, ckpt

def setup_logger(experiment_log_dir):
    log_path = os.path.join(experiment_log_dir, f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}")
    return LoggerManager(log_path).logger

def setup_tensorboard(experiment_log_dir):
    return TensorBoardManager(experiment_log_dir).writer

def setup_env(config, args: argparse.Namespace, ckpt=None):
    """
    Env 세팅
    """
    env = gym.make(args.env)

    if ckpt is None:
        env_config = {}
        if "discrete" in str(env.action_space.__class__).lower():
            env_config["action_type"] = "discrete"
            env_config["action_dim"] = env.action_space.n.item()

        if "box" in str(env.observation_space.__class__).lower():
            # Continous space
            env_config["state_dim"] = env.observation_space.shape[0]
        
        # if args.wrapper is not None:
        #     # import class env and apply env = wrapper(env)
        #     env_config["wrapper"] = args.wrapper
        config["env"] = env_config
    else:
        # env도 config에 들어있는 wrapper 사용하도록? 구성
        pass
    
    return env

def setup_buffer(config: dict, args: argparse.Namespace, ckpt=None):
    """
    Buffer set up
    todo) checkpoint에 buffer안에 지금까지 들어있는 데이터 저장해야 하나? 과하나?

    ckpt: 이후 확장성, 다른 setup 함수와의 일관성을 위해 존재.
    """
    
    buffer_class = import_class(f"{args.buffer_class_module}.{args.data_class}")
    buffer = buffer_class(batch_size=args.batch_size, device=args.device, config=config, args=args)

    return buffer

def setup_networks_and_agent(config: dict, args: argparse.Namespace, ckpt=None):
    """
    Network와 agent setup
    """
    agent_class = get_agent_class(args)
    networks = agent_class.setup_networks(config)
    agent = agent_class(args=args, config=config, **networks)

    if ckpt is not None:
        agent.load_state_dict(ckpt)

    return networks, agent

def get_agent_class(args: argparse.Namespace):
    agent_class_module = get_agent_class_module(args)
    agent_class = import_class(f"{agent_class_module}.{args.agent}")

    return agent_class
            
def get_class_module_names(args: argparse.Namespace):
    """
    buffer, agent, trainer 클래스 모듈 임포
    """
    buffer_class_module = get_buffer_class_module(args)
    agent_class_module = get_agent_class_module(args)
    trainer_class_module  = get_trainer_class_module(args)

    return buffer_class_module, agent_class_module, trainer_class_module

def get_agent_class_module(args: argparse.Namespace):
    agent_category = get_agent_category(args)
    return f"{args.training_mode}.{agent_category}.agents"

def get_agent_category(args: argparse.Namespace):
    """
    agent class 이름을 받아서 알맞은 agent 카테고리(training_mode) 반환
    """
    categories = {
        "actor_critic": ["sac"]
    }
    for category, keywards in categories.items():
        for keyward in keywards:
            if keyward.lower() in args.agent.lower():
                return category

def get_buffer_class_module(args):
    return f"{args.training_mode}.buffers"

def get_trainer_class_module(args):
    return f"{args.training_mode}.trainers"
         


