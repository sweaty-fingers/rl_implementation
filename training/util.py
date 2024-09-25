import argparse
import importlib
import gymansium_tutorials as gym

def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'upgradable_dehy.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def get_class_module_name(training_mode: str):
    network_class_module = f"{training_mode}.networks"
    trainer_class_module = f"{training_mode}.trainers"
    buffer_class_module = f"{training_mode}.buffers"

    return network_class_module, trainer_class_module, buffer_class_module

def setup_buffer_and_network(config: dict, args: argparse.Namespace):
    # agent에 따라 module 경로 수정하는 코드 작성하기
    buffer_class = import_class(f"{args.buffer_class_module}.{args.data_class}")
    network_class = import_class(f"{args.network_class_module}.{args.model_class}")
    
    buffer = buffer_class(batch_size=args.batch_size, device=args.device, args=args)
    network = network_class(data_config=data.config(), args=args)

    return data, network

def log_outputs(outputs: dict, stage: str, logger = None):
    s = f"Best {stage} Performance:"
    for key, value in outputs.items():
        s += f"{key} = {value:.4f} | "
    logger.info(s)

def get_env(env_name, wrapper=None):
    env, env_config = gym.make(env_name), {}

    if "discrete" in str(env.action_space.__class__).lower():
        env_config["action_type"] = "discrete"
        env_config["action_dim"] = env.action_space.n.item()

    if "box" in str(env.observation_space.__class__).lower():
        # Continous space
        env_config["state_dim"] = env.observation_space.shape[0]
    
    if wrapper is not None:
        env = wrapper(env)
    
    return env, env_config

