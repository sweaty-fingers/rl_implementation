import argparse
import importlib

def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'upgradable_dehy.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def get_class_module_name(training_mode: str):
    network_class_module = f"{training_mode}.networks"
    trainer_class_module = f"{training_mode}.trainers"

    return network_class_module, trainer_class_module

def setup_data_and_network_from_args(args: argparse.Namespace):
    data_class = import_class(f"{args.data_class_module}.{args.data_class}")
    network_class = import_class(f"{args.network_class_module}.{args.model_class}")
    
    data = data_class(args=args)
    network = network_class(data_config=data.config(), args=args)

    return data, network

def log_outputs(outputs: dict, stage: str, logger = None):
    s = f"Best {stage} Performance:"
    for key, value in outputs.items():
        s += f"{key} = {value:.4f} | "
    logger.info(s)
