import argparse
import os, sys
import glob
import torch

from utilities.util import set_seed, create_directory, save_dict_to_json, remove_old_checkpoints, get_config
from training.util import import_class, get_class_module_names, setup_data_and_network_from_args, log_outputs, get_env
from training.setup import setup_experiment_log_dir, setup_logger, setup_tensorboard, setup_env, setup_networks_and_agent, setup_buffer
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
    parser.add_argument('--seed', default=SEED, type=int, help='seed value')
    parser.add_argument(
        "--env",
        type=str,
        default="",
        help="String identifier for environment.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="",
        help="String identifier for the model class, relative to {training_mode}.trainers",
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default="ReplayBuffer",
        help="Replay buffer for training"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )

    # 사용할 data, model 클래스를 임포트한 후 
    temp_args, _ = parser.parse_known_args()

    buffer_class_module, agent_class_module, trainer_class_module = get_class_module_names(temp_args.training_mode)

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

        ## Logger & Tensorboard setting
        logger = setup_logger(experiment_log_dir=experiment_log_dir)
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

        config["DATA_ATTRS"] = data.get_attrs()
        config["MODEL_ATTRS"] = model.get_attrs()
        config["TRAINER_ATTRS"] = trainer.get_attrs()
        
        remove_old_checkpoints(config["CHECKPOINTS"]["CONFIG_DIRNAME"], prefix="config", extension=".json")
        save_dict_to_json(config, os.path.join(config["CHECKPOINTS"]["CONFIG_DIRNAME"], "config.json"), readonly=True)

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
        
        outputs = trainer.fit(data, args.max_epochs)
        logger.info(f"{Fore.BLUE}\nTraining end, final epoch:  {outputs['epoch']} | best epoch: {trainer.best_state['epoch']}{Style.RESET_ALL}")

        for stage in ["train", "valid"]:
            log_outputs(outputs=outputs[stage], stage=stage, logger=logger)
            
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