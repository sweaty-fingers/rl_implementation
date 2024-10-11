import argparse
import os, sys
import glob
import torch

from utilities.util import set_seed, set_device, save_dict_to_json, remove_old_checkpoints
from utilities.logger_util import set_log_level
from training.setup import setup_experiment_log_dir, setup_logger, setup_tensorboard, setup_env, setup_networks_and_agent, setup_buffer, import_class, get_class_module_names
from colorama import Fore, Style
from datetime import datetime

ENV_CLASS_MODULE = "envs"
SEED = 2024

start_time = datetime.now()

def _setup_parser():
    parser = argparse.ArgumentParser(add_help=False)
    
    # 훈련에 필요한 기본 argument
    # parser.add_argument(
    #     "--profile",
    #     action="store_true",
    #     default=False,
    #     help="If passed, uses the PyTorch Profiler to track computation, exported as a Chrome-style trace.",
    # )
    parser.add_argument("--training_mode", default="online_training.off_policy")
    parser.add_argument("--print_log_level", type=str, default="info", help="Logger level for printing cli")
    parser.add_argument('--seed', default=SEED, type=int, help='seed value')
    parser.add_argument('--config', default=None, type=str, help="Path of config file to run experiment")
    parser.add_argument(
            "--gpus",
            default=None,
            type=int,
            help="id(s) for GPU_VISIBLE_DEVICES(MPS or CUDA)",
        )
    parser.add_argument(
        "--max_steps", type=int, default=1, help="Max steps to update agent"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        help="String identifier for environment.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="DiscreteSACAgent",
        help="String identifier for the model class, relative to {training_mode}.agents",
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default="ReplayBuffer",
        help="Replay buffer for training"
    )
    parser.add_argument(
        "--networks",
        type=str,
        default="",
        help="Replay buffer for training"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="BaseTrainer",
        help="String identifier for the trainer class, relative to {training_mode}.trainers"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )

    # 사용할 data, model 클래스를 임포트한 후 
    temp_args, _ = parser.parse_known_args()

    buffer_class_module, agent_class_module, trainer_class_module = get_class_module_names(temp_args)

    print("Data and model loaded ...")
    buffer_class = import_class(f"{buffer_class_module}.{temp_args.buffer}")
    agent_class = import_class(f"{agent_class_module}.{temp_args.agent}")
    trainer_class = import_class(f"{trainer_class_module}.{temp_args.trainer}")

    # Get model, and LitModel specific arguments

    buffer_group = parser.add_argument_group("Buffer Args")
    buffer_class.add_to_argparse(buffer_group)

    agent_group = parser.add_argument_group("Agent Args")
    agent_class.add_to_argparse(agent_group)

    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_class.add_to_argparse(trainer_group)

    parser.add_argument("--help", "-h", action="help")
    
    return parser

def main():
    try: 
        parser = _setup_parser()
        args = parser.parse_args()
        # experiment_log_dir setting
        experiment_log_dir, config, args, ckpt = setup_experiment_log_dir(args)
        set_seed(args.seed)
        args.device = set_device(args.gpus)
        config["device"] = args.device
        
        ## Logger & Tensorboard setting
        logger = setup_logger(experiment_log_dir=experiment_log_dir)
        set_log_level(args.print_log_level)
        writer = setup_tensorboard(experiment_log_dir=experiment_log_dir)

        # Set up envs
        env = setup_env(config=config, args=args, ckpt=ckpt)

        # Set up buffer
        buffer = setup_buffer(config=config, args=args, ckpt=ckpt)

        # Set up networks
        networks, agent = setup_networks_and_agent(config=config, args=args, ckpt=ckpt)
        logger.debug("Data and model loaded ...")

        # trainer setting
        trainer_class = import_class(f"{args.trainer_class_module}.{args.trainer_class}")
        trainer = trainer_class(env=env, agent=agent, buffer=buffer, args=args, checkpoint=ckpt)
        
        if trainer.checkpoint is not None:
            trainer.load_state()

        config["buffer"] = buffer.config
        networks_config = {}
        for key, network in networks.items():
            networks_config[key] = network.config
        config["networks"] = networks_config
        config["agent"] = agent.config
        config["trainer"] = trainer.config
        
        remove_old_checkpoints(config["checkpoint"]["dirs"]["config"], prefix="config", extension=".json")
        save_dict_to_json(config, os.path.join(config["checkpoint"]["dirs"]["config"], "config.json"), readonly=True)

        logger.debug("set_up trainer...")
        
        logger.info("=" * 45)
        logger.info(f'{Fore.GREEN}Training dataset: {args.data_class}{Style.RESET_ALL}')
        logger.info(f'{Fore.GREEN}Batch_size: {args.batch_size}{Style.RESET_ALL}')
        logger.info(f'{Fore.GREEN}Training model: {args.model_class}{Style.RESET_ALL}')
        logger.info(f"{Fore.GREEN}Training trainer: {args.trainer_class}{Style.RESET_ALL}")
        logger.info(f'{Fore.GREEN}Mode: {args.training_mode}{Style.RESET_ALL}')
        logger.info(f"{Fore.GREEN}Device: {args.trainer.device}{Style.RESET_ALL}")
        
        logger.debug(f"Args:    {args._get_kwargs()}\n" )
        logger.debug(f"input command: \n{'python ' + ' '.join(sys.argv)}") 
        logger.debug("=" * 45)
        logger.debug("\n")

        logger.info(f"{Fore.BLUE}Training start{Style.RESET_ALL}")
        
        outputs = trainer.run_episodes()
        logger.info(f"{Fore.BLUE}\nTraining end, final epoch:  {outputs['epoch']} | best epoch: {trainer.best_state['epoch']}{Style.RESET_ALL}")
            
        # outputs = trainer.test(data)
        # stage = "test"
        # log_outputs(outputs=outputs[stage], stage=stage, logger=logger)
        
        writer.flush()
        writer.close()
        
        logger.info(f"Training time is : {datetime.now()-start_time}")
        # logger.debug("#"* 25)
        #logger.debug("data_args:")
        #logger.debug(f"{data.__dict__}")

    except Exception as e:
        raise e

if __name__ == "__main__":
    main()