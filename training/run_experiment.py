import argparse
import os, sys
import glob
import torch

from training.util import import_class, get_class_module_name, setup_data_and_network_from_args, log_outputs, get_env
from utilities.util import set_seed, create_directory, save_dict_to_json, remove_old_checkpoints, get_config
from utilities.managers import ConfigManager, TensorBoardManager, LoggerManager
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
    parser.add_argument("--training_mode", default="online_training")
    parser.add_argument("--max_epochs", default=2, type=int, help='max_epochs')
    parser.add_argument('--seed', default=SEED, type=int, help='seed value')
    parser.add_argument(
        "--env",
        type=str,
        default="InvertPendulum",
        help="String identifier for environment.",
    )
    parser.add_argument(
        "--network",
        type=str,
        default="DistFCNN",
        help="String identifier for the model class relatvie to {training_mode}.networks.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="BaseOfflineTrainer",
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

    network_class_module, trainer_class_module, buffer_class_module = get_class_module_name(temp_args.training_mode)

    print("Data and model loaded ...")
    buffer_class = import_class(f"{buffer_class_module}.{temp_args.buffer}")
    model_class = import_class(f"{network_class_module}.{temp_args.network}")
    trainer_class = import_class(f"{trainer_class_module}.{temp_args.trainer}")

    # Get model, and LitModel specific arguments
    model_group = parser.add_argument_group("Network Args")
    model_class.add_to_argparse(model_group)

    buffer_group = parser.add_argument_group("Buffer Args")
    buffer_class.add_to_argparse(buffer_group)

    trainer_group = parser.add_argument_group("Trainer Args")
    trainer_class.add_to_argparse(trainer_group)

    parser.add_argument("--help", "-h", action="help")
    
    return parser

def main():
    try: 
        experiment_log_dir = None
        ckpt = None

        parser = _setup_parser()
        args = parser.parse_args()

        set_seed(args.seed)
        
        # Checkpoint setting
        if args.checkpoint is not None:
            experiment_log_dir = args.checkpoint
            args.config = os.path.join(experiment_log_dir, "configs", "config.json")
            config_manager = ConfigManager(args.config)

            try:
                pattern = os.path.join(config_manager.config["CHECKPOINTS"]["CKPT_DIRNAME"], "model_best*.pth")
                ckpt_path = glob.glob(pattern)[0]
                ckpt = torch.load(ckpt_path)

            except FileExistsError as e:
                    print(f"Error: no checkpoint directory found! {e}")

            if args.max_epochs <= ckpt["epoch"] + 1:
                raise ValueError(f"max_epochs should be larger than current epoch {ckpt['epoch']}")
            
        else:
            config_manager = ConfigManager(args.config)
            logs_save_dir = config_manager.config["DIRS"]["ROOT"]["EXPERIMENT_DIRNAME"]
            experiment_description = f"{args.env_class}/{args.trainer_class}/{args.model_class}"

            if args.max_epochs < 3:
                experiment_log_dir = os.path.join(logs_save_dir, experiment_description, "test")
            else:
                i = 1
                while True: 
                    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, f"run{i}")
                
                    if not os.path.exists(os.path.join(experiment_log_dir, "csv")):
                        break
                    i += 1
                    
            checkpoints = {}
            checkpoints["ROOT_DIRNAME"] = experiment_log_dir
            checkpoints["CONFIG_DIRNAME"] = os.path.join(experiment_log_dir, "configs")
            checkpoints["CSV_DIRNAME"] = os.path.join(experiment_log_dir, "csv")
            checkpoints["CKPT_DIRNAME"] = os.path.join(experiment_log_dir, "ckpt")

            for _, dir in checkpoints.items():
                create_directory(dir)
                
            config_manager.config["CHECKPOINTS"] = checkpoints

        # Logger & Tensorboard setting
        log_path = os.path.join(experiment_log_dir, f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}")
        logger_manager = LoggerManager(log_path)

        tensorboard_manager = TensorBoardManager(experiment_log_dir)
        writer = tensorboard_manager.get_writer()

        # Set up envs
        env, env_config = get_env(env_name=args.env), {}
        config_manager.config["env"] = env_config

        # Set up agent & network 
        agent, model = setup_data_and_network_from_args(args)
        data_prepare_and_setup(data)
        logger.debug("Data and model loaded ...")
        


        # trainer setting
        trainer_class = import_class(f"{args.trainer_class_module}.{args.trainer_class}")        
        trainer = trainer_class(model=model, args=args, checkpoint=ckpt)
        
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